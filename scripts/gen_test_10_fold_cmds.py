import os
for i in range(10):
  arch = 'resnet101' #'inception_v3' #'resnet50'
  exp_name = '{}_fold_{}'.format(arch, i+1)
  cmd = "CUDA_VISIBLE_DEVICES=0 python make_submissions.py "\
        " --arch {1}  --model_file ckpt/{2}/model_epoch_15.pth --ten_crop "\
        " --extra_name fold_{0}_tencrop ".format(i+1, arch, exp_name)
  
  print('\n', cmd)
  os.system(cmd)
