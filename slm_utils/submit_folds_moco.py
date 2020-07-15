import subprocess
import shlex
import os 

# Notes: This is the setup used to get the 5 folds of the rotnet for our evaluation of rotation predictions

# TODO: you will want to change this to your checkpoint filepath. 
checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'

# Imagenet pretrain
for fold in range(5): # TODO change. 

    # This is all code for my queue system, basically just submits it and writes the output to a txt file. 
    filename = '/userdata/smetzger/all_deepul_files/runs/imnet_kfolds_%d.txt' %fold
    string = "submit_job -q mind-gpu"
    string += " -m 318 -g 4"
    string += " -o " + filename
    string += ' -n inet_moco'
    string += ' -x python /userdata/smetzger/all_deepul_files/deepul_proj/moco/main_moco.py'

    # add all the default args: 
    string += " -a resnet50 --lr 0.015  --batch-size 128 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1"
    string += ' --moco-t 0.2' # MoCov2 arguments. 
    string += ' --checkpoint_fp ' + str(checkpoint_fp)
    string += ' --rank 0'
    string += " --data /userdata/smetzger/data/imagenet/imagenet12/"
    string += ' --reduced_imgnet'
    string += ' --custom_aug_name rrc_pure'
    string += ' --mlp --cos --epochs 500' # because we have the reduced dataset, we run for 500 epochs
        # string += ' --rand_resize_only'
    string += ' --kfold %d' %fold

    cmd = shlex.split(string)
    print(cmd)
    subprocess.run(cmd, stderr=subprocess.STDOUT)
