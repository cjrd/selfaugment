print('Sko Buffs')
import subprocess
import shlex
import os 

base_model_name = ''
epochs = 750
import os
checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'

# For resuming 

def find_model(name, fold, epochs, basepath="/userdata/smetzger/all_deepul_files/ckpts"):
    """
    name = model name
    fold = which fold of the data to find. 
    epochs = how many epochs to load the checkpoint at (e.g. 750)
    
    """
    for file in os.listdir(basepath):
        print(file)
        if name in str(file) and 'fold_%d' %fold in str(file):
            print(file)
            if str(file).endswith(str(epochs-1) + '.tar') or str(file).endswith(str(epochs) + '.tar'): 
                return os.path.join(basepath, file)
            
    print("COULDNT FIND MODEL")
    assert True==False # just throw and error. 

base_name = '750epochs_512bsz_0.4000lr_mlp_cos_rotnet'
# Notes: This is the setup used to get the 5 folds of the rotnet for our evaluation of rotation predictions

# for fold in range (5): 
filename = '/userdata/smetzger/all_deepul_files/runs/svhn_mocov2_reference.txt'
string = "submit_job -q mind-gpu"
string += " -m 318 -g 4"
string += " -o " + filename
string += ' -n svhn'
string += ' -x python /userdata/smetzger/all_deepul_files/deepul_proj/moco/main_moco.py'

# add all the default args: 
string += " -a resnet50 --lr 0.4  --batch-size 512 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1"
string += ' --moco-t 0.2' # MoCov2 arguments. 
string += ' --checkpoint_fp ' + str(checkpoint_fp)
string += ' --rank 0'
string += " --data /userdata/smetzger/data/cifar_10/ --notes 'svhn reference'"


# THIS LINE IS HUGE: TRAIN THE ROTNET HEAD.
# string += ' --rotnet --nomoco' # We are only training rotnets. 
string += ' --rand_resize_only'
string += ' --resume ' + checkpoint_fp+ '/3ug6V_100epochs_512bsz_0.4000lr_mlp_cos_custom_aug_rrc_puresvhn_0099.tar'
string += ' --start-epoch 100'
# string += ' --aug-plus'
string += ' --dataid svhn'
string += ' --mlp --cos --epochs 750'
string += ' --checkpoint-interval 250'

cmd = shlex.split(string)
print(cmd)
subprocess.run(cmd, stderr=subprocess.STDOUT)