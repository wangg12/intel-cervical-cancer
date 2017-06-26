import os
for i in range(10):
  arch = 'resnet101' #'inception_v3' #'resnet101' # 'vgg16' #, 'resnet152'
  cmd = "CUDA_VISIBLE_DEVICES=1 python main.py --pretrained 1  --pin_memory 0 --workers 8 "\
  " --manual_seed 1438 "\
  " --n_epoch 25 "\
  " --ckpt_epoch 5 "\
  " --optimizer nag "\
  " --lr 0.001 "\
  " --lr_decay_epoch 10 "\
  " --lr_decay_factor 0.1 "\
  " --batch_size 16 "\
  " --exp_name {1}_retrain_fold_{0} "\
  " --arch {1} "\
  " --ckpt_dir ckpt/{1}_retrain_fold_{0} "\
  " --CE_Loss "\
  " --train_idx ./data_utils/train_idx_{0}.csv "\
  " --val_idx ./data_utils/val_idx_{0}.csv".format(i+1, arch)
  print("\n", cmd)
  os.system(cmd)

