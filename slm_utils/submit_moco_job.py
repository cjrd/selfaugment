print('Sko Buffs')
import subprocess
import shlex
import os 


checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'
  # elif name == 'rrc_min_max': 
  #       transform_train.transforms.insert(0, Augmentation(fa_rrc_min_max()))
  #   elif name == 'rrc_min_max_weighted': 
  #       transform_train.transforms.insert(0, Augmentation(fa_rrc_min_max_weighted()))
  #   elif name == 'rrc_pure': 
  #       print('not adding anything')
  #   elif name == 'rrc_max_icl_top2': 
  #       transform_train.transforms.insert(0, Augmentation(fa_rrc_max_icl_top2()))
  #   elif name == 'rrc_min_rot_top2': 
  #       transform_train.transforms.insert(0, Augmentation(fa_rrc_min_rot_top2()))
  #   elif name == 'rrc_min_max_top2': 
  #       transform_train.transforms.insert(0, Augmentation(fa_rrc_min_max_top2()))
custom_aug_names = ['rrc_pure']

for custom_aug_name in custom_aug_names: 
 
	filename = '/userdata/smetzger/all_deepul_files/runs/moco_baseline_augs_fresh_restarted_fuq.txt'
	string = "submit_job -q mind-gpu"
	string += " -m 318 -g 4"
	string += " -o " + filename
	string += ' -n MoCo1'
	string += ' -x python /userdata/smetzger/all_deepul_files/deepul_proj/moco/main_moco.py'

	# add all the default args: 
	string += " -a resnet50 --lr 0.4  --batch-size 512 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1"
	string += ' --moco-t 0.2' # MoCov2 arguments. 
	string += ' --checkpoint_fp ' + str(checkpoint_fp)
	string += ' --rank 0'
	string += " --data /userdata/smetzger/data/cifar_10/ --notes 'fresh_750_SVHN_mocov2'"

	string += ' --mlp --cos --epochs 750'

	string += ' --checkpoint-interval 250'
	string += ' --dataid svhn'


	# # HUGE LINE: only use rand_resize_crop as the base xform.
	string += ' --aug-plus'

	cmd = shlex.split(string)
	print(cmd)
	subprocess.run(cmd, stderr=subprocess.STDOUT)