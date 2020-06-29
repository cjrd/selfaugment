print('Sko Buffs')
import subprocess
import shlex
import os 
import time

def find_model(name, epochs, basepath="/userdata/smetzger/all_deepul_files/ckpts"):
    """
    name = model name
    fold = which fold of the data to find. 
    epochs = how many epochs to load the checkpoint at (e.g. 750)
    
    """

    path_list = []
    file_list = []
    for file in os.listdir(basepath):
        if name in str(file):
            if str(file).endswith(str(epochs-1) + '.tar'): 
                path_list.append(os.path.join(basepath, file))
                file_list.append(file)
            
    return path_list, file_list

mind_list = [1, 2, 5, 6, 8]
mind_list = mind_list*10

iii = 0 

for custom_aug in ['single_aug_study']:

    base_name = '10epochs_128bsz_0.0150lr_mlp_cos_custom_aug_single_aug_studyimagenet_0009'
    # base_name = '100epochs_512bsz_0.4000lr_mlp_cos_custom_aug_' + custom_aug + 'svhn'
    # CaUWi_100epochs_512bsz_0.4000lr_mlp_cos_custom_aug_single_aug_studysvhn_0099
    print(base_name)
    checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'

    models, files = find_model(base_name, 10)

    print(models)
    print(len(models))

    for task in ['rotation']: 

        for model, name in zip(models, files): 
            filename = '/userdata/smetzger/all_deepul_files/runs/lincls_' + name[:-4] + '_rotation.txt'
            string = "submit_job -q mind-gpu@mind%d" %(mind_list[iii])
            string += " -m 318 -g 4"
            string += " -o " + filename
            string += ' -n lincls'
            string += ' -x python /userdata/smetzger/all_deepul_files/deepul_proj/moco/main_lincls.py'

            # add all the default args: 
            string += " -a resnet50 --lr 0.5  --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1"
            string += ' --checkpoint_fp ' + str(checkpoint_fp)
            string += ' --rank 0'
            string += ' --pretrained ' + model
            string += " --data /userdata/smetzger/data/imagenet/imagenet12/  --notes 'training_single_aug'"
            string += " --task " + task
            string += " --schedule 5 8 --epochs 13"
            string += " --dataid imagenet"

            cmd = shlex.split(string)
            print(cmd)
            import subprocess
            subprocess.run(cmd, stderr=subprocess.STDOUT)

            iii += 1
            print(iii)
            print()