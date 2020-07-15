print('Sko Buffs')
import subprocess
import shlex
import os 


checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'

custom_aug_names = ['single_aug_study']
for idx in range(15): 
	for custom_aug_name in custom_aug_names: 
	
		filename = '/userdata/smetzger/all_deepul_files/runs/imagenet_' + custom_aug_name + '_single_aug_test_%d.txt' %(idx)
		string = "submit_job -q mind-gpu"
		string += " -m 318 -g 4"
		string += " -o " + filename
		string += ' -n MoCo1'
		string += ' -x python /userdata/smetzger/all_deepul_files/deepul_proj/moco/main_moco.py'

		# add all the default args: 
		string += " -a resnet50 --lr 0.015  --batch-size 128 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1"
		string += ' --moco-t 0.2' # MoCov2 arguments. 
		string += ' --checkpoint_fp ' + str(checkpoint_fp)
		string += ' --rank 0'
		string += " --data /userdata/smetzger/data/imagenet/imagenet12/  --notes 'imgnet_single_aug_%d'" %idx

		string += ' --mlp --cos --epochs 100'

		# Huge line here, submit custom agumentations: 
		string += ' --custom_aug_name ' + custom_aug_name
		string += ' --single_aug_idx ' + str(idx)
		string += ' --dataid imagenet'
		string += ' --kfold 0'
		string += ' --reduced_imgnet'


		# HUGE LINE: only use rand_resize_crop as the base xform.
		# string += ' --rand_resize_only'
		# string += ' --resume ' + checkpoint_fp + '/SsAyL_2000epochs_512bsz_0.4000lr_mlp_augplus_cos_1750.tar'
		# string += ' --start-epoch 1750'

		cmd = shlex.split(string)
		print(cmd)
		subprocess.run(cmd, stderr=subprocess.STDOUT)


custom_aug_names = ['rrc_pure']
idx = 33
for custom_aug_name in custom_aug_names: 


	filename = '/userdata/smetzger/all_deepul_files/runs/imagenet_' + custom_aug_name + '_single_aug_test_%d.txt' %(idx)
	string = "submit_job -q mind-gpu"
	string += " -m 318 -g 4"
	string += " -o " + filename
	string += ' -n MoCo1'
	string += ' -x python /userdata/smetzger/all_deepul_files/deepul_proj/moco/main_moco.py'

	# add all the default args: 
	string += " -a resnet50 --lr 0.015  --batch-size 128 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1"
	string += ' --moco-t 0.2' # MoCov2 arguments. 
	string += ' --checkpoint_fp ' + str(checkpoint_fp)
	string += ' --rank 0'
	string += " --data /userdata/smetzger/data/imagenet/imagenet12/  --notes 'imgnet_single_aug_%d'" %idx

	string += ' --mlp --cos --epochs 100'

	# Huge line here, submit custom agumentations: 
	string += ' --custom_aug_name ' + custom_aug_name
	string += ' --dataid imagenet'
	string += ' --kfold 0'
	string += ' --reduced_imgnet'


	# HUGE LINE: only use rand_resize_crop as the base xform.
	# string += ' --rand_resize_only'
	# string += ' --resume ' + checkpoint_fp + '/SsAyL_2000epochs_512bsz_0.4000lr_mlp_augplus_cos_1750.tar'
	# string += ' --start-epoch 1750'

	cmd = shlex.split(string)
	print(cmd)
	subprocess.run(cmd, stderr=subprocess.STDOUT)
