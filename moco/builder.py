# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from itertools import chain


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, K=65536, m=0.999, T=0.07, mlp=False, dataid="cifar10", multitask_heads={}):
        """
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        
        # create the encoders
        # num_classes does not matter here
        encoder = base_encoder(num_classes=128)

        if dataid =="cifar10" or dataid =='svhn': 
            # use the layer the SIMCLR authors used for cifar10 input conv, checked all padding/strides too. 
            encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False) 
            # Get rid of maxpool, as in SIMCLR cifar10 experiments. 
            encoder.maxpool = nn.Identity()

        # what dimension should we make the heads?
        print(encoder)
        dim_mlp = encoder.fc.weight.shape[1]

        # hack to "remove" the fc layer
        encoder.fc = nn.Identity()
        modules = {
            # pop off the final fc layer and store the shared encoder
            "encoder": encoder,
        }

        # Multitask is easy: we just need lots of [mlp/linear] heads
        for mt in multitask_heads.keys():
            fc = nn.Linear(dim_mlp, multitask_heads[mt]["num_classes"])
            if mlp:
                fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), fc)
            modules[mt] = fc
            
        self.model = nn.ModuleDict(modules)

        # Treat moco special, since it needs the key model
        if "moco" in multitask_heads:
            mocodim=multitask_heads["moco"]["num_classes"]
            self.encoder_k = base_encoder(num_classes=mocodim)

            # now apply the same encoder updates
            if mlp:
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
                
            if dataid=="cifar10" or dataid == "svhn":
                self.encoder_k.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False)
                self.encoder_k.maxpool = nn.Identity()
                
            # set the params to be the same starting values
            
            for param_q, param_k in zip(self._get_moco_params(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  
                param_k.requires_grad = False  # not update by gradient, update by momentum technique

            # create the moco queue
            self.register_buffer("queue", torch.randn(mocodim, K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    def _get_moco_params(self):
        return chain(self.model["encoder"].parameters(), self.model["moco"].parameters())

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self._get_moco_params(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, head, im_q, im_k=None, evaluate=False):
        if head=="moco":
            return self.moco_forward(im_q, im_k, evaluate)
        elif head=="rotnet":
            return self.model[head](self.model["encoder"](im_q))
        else:
            raise NotImplementedError("The following head has not been implemented: {}".forward(head))
        
    
    def moco_forward(self, im_q, im_k, evaluate=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        
        # compute query features

        q = self.model["encoder"](im_q)
        q = self.model["moco"](q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if not evaluate: 
            # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)


            if not evaluate: 
            # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        if not evaluate: 
        # dequeue and enqueue
            self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
