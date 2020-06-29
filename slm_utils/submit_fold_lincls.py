print('Sko Buffs')
import subprocess
import shlex
import os 

base_model_name = ''
epochs = 750
import os
def find_model(name, fold, epochs, basepath="/userdata/smetzger/all_deepul_files/ckpts"):
    """
    name = model name
    fold = which fold of the data to find. 
    epochs = how many epochs to load the checkpoint at (e.g. 750)
    
    """

    path_list = []

    for file in os.listdir(basepath):
        if name in str(file):
            if str(file).endswith(str(epochs-1) + '.tar'): 
                if 'fold_%d' %(fold) in file: 
                    return (os.path.join(basepath, file))
            
    print("COULDNT FIND MODEL")
    assert True==False # just throw and error. 


checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'
epochs = 500

i = 0

mind_list = [2, 1, 8, 5, 6]

names = ['7VThG', '46PlZ', '9NOo9', '9Odci', '0jG5u' ]

for task in ['rotation']: 
    for fold in range (5): # TOD CHANGE BACK TO 5 


        # Stuff for my queue system
        filename = '/userdata/smetzger/all_deepul_files/runs/lincls_new_' + 'logos' + '_fold_%d' %fold + '_' + task + '_kfold.txt'
        string = "submit_job -q mind-gpu@mind%d" %mind_list[i]
        string += " -m 318 -g 4"
        string += " -o " + filename
        string += ' -n lincls'
        string += ' -x python /userdata/smetzger/all_deepul_files/deepul_proj/moco/main_lincls.py'

        # add all the default args: 
        string += " -a resnet50 --lr 0.5  --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1"
        string += ' --checkpoint_fp ' + str(checkpoint_fp)
        string += ' --rank 0'
        string += " --data /userdata/smetzger/data/logos/train_and_test/ --notes 'training_rotnet'"
        string += " --task " + task


        string += " --schedule 10 20 --epochs 50"
        string += " --dataid logos"
        string += " --reduced_imgnet"
        string += " --kfold %d" %fold
        # string += " --newid"

        # All your checkpoints will have this in the filename, so then you can find them to use as the pretrained model. 
        base_name = '500epochs_128bsz_0.0150lr_mlp_cos_fold_%dlogos_0499' %fold

        print(base_name)
        string += ' --pretrained ' + checkpoint_fp + '/' + names[i] + '_' + base_name + '.tar'

        cmd = shlex.split(string)
        print(cmd)
        i+=1
        print(i)
        subprocess.run(cmd, stderr=subprocess.STDOUT)