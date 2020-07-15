import subprocess
import shlex
import os 

# TODO: you will want to change this to your checkpoint filepath. 
checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'

# Imagenet pretrain

faa_names = ['imagenet_minmax', 'imagenet_minmax_weighted', 'imagenet_min_rotation', 'imagenet_min_icl', 'imagenet_max_icl']
for faa_name in faa_names:  # TODO change. 

    print(faa_name)

    # This is all code for my queue system, basically just submits it and writes the output to a txt file. 
    filename = '/userdata/smetzger/all_deepul_files/runs/logos_%s.txt' %(faa_name)
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
    string += " --data /path/to/imagenet/"
    string += " --notes 'imagenet'"# %fold
    string += ' --dataid imagenet'
    string += ' --custom_aug_name ' + faa_name
    string += ' --mlp --cos --epochs 100' 
    cmd = shlex.split(string)
    print(cmd)
    subprocess.run(cmd, stderr=subprocess.STDOUT)
