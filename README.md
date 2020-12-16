# SelfAugment

[Paper](https://arxiv.org/abs/2009.07724)

```
@misc{reed2020selfaugment,
      title={SelfAugment: Automatic Augmentation Policies for Self-Supervised Learning}, 
      author={Colorado Reed and Sean Metzger and Aravind Srinivas and Trevor Darrell and Kurt Keutzer},
      year={2020},
      eprint={2009.07724},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

SelfAugment extends [MoCo](https://github.com/facebookresearch/moco) to include automatic unsupervised augmentation selection.
In addition, we've included the ability to pretrain on several new datasets and included a [wandb](http://wandb.ai/) integration.

# Using your own dataset. 
To interface your own dataset, make sure that you carefully check the three main scripts to incorporate your dataset: 
1. main_moco.py
2. main_lincls.py
3. faa.py 

Some things to check: 
1. Ensure that the sizing for your dataset is right. If your images are 32x32 (e.g. CIFAR10) - you should ensure that you are using the CIFAR10 style model, which uses a 3x3 input conv, and resizes images to be 28x28 instead of 224x224 (e.g. for ImageNet). This can make a big difference! 
2. If you want selfaugment to run quickly, consider using a small subset of your full dataset. For example, for ImageNet, we only use a small subset of the data - 50,000 random images. This may mean that you need to run unsupervised pretraining for longer than you usually do. We usually scale the number of epochs MoCov2 runs so that the number of total iterations is the same, or a bit smaller, for the subset and the full dataset. 

# Base augmentation. 
If you want to find the base augmentation, then use slm_utils/submit_single_augmentations.py

This will result in 16 models, each with the results of self supervised training using ONLY the augmentation provided.
slm_utils/submit_single_augmentations is currently using imagenet, so it uses a subset for this part.

Then you will need to train rotation classifiers for each model. this can be done using main_lincls.py

# Train 5 Folds of MoCov2 on the folds of your data. 
To get started, train 5 moco models using only the base augmentation. 
To do this, you can run python slm_utils/submit_moco_folds.py.

# Run SelfAug
Now, you must run SelfAug on your dataset. Note - some changes in dataloaders may be necessary depending on your dataset. 

@Colorado, working on making this process cleaner. 

For now, you will need to go into faa_search_single_aug_minmax_w.py, and edit the config there. I will change this to argparse here soon.
The most critical part of this is entering your checkpoint names in order of each fold under config.checkpoints. 

Loss can be rotation, icl, icl_and_rotation.
If you are doing icl_and_rotation, then you will need to normalize the loss_weights in loss_weight dict so that each loss is 1/(avg loss across k-folds) for each type of loss, I would just use the loss that was in wandb (rot train loss, and ICL loss from pretraining). Finally, you are trying to maximize negative loss with Ray, so a negative weighting in the loss weights means that the loss with that weight will be maximized. 

# Retrain using new augmentations found by SelfAug. 

Just make sure to change the augmentation path to the pickle file with your new augmentations in load_policies function in get_faa_transforms.py
Then, submit the job using slm_utils/submit_faa_moco.py





