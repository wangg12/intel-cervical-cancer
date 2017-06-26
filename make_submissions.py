# -*- coding: utf-8 -*-
''' python 3.4+
author: Yu Yang, Gu Wang, Shi Yan
'''
from __future__ import division, absolute_import, print_function
import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# to solve the default PIL loader problem in torchvision

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torchvision import models, transforms
# import torchsample as ts
# from torch.utils.data import DataLoader

from data_utils.my_folder import MyImageFolder
from utils import weights_init, get_augmented_test_set, KaggleLogLoss, get_multi_scale_crop_test_set
from models import nets

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_root', default='./data/resized_data/', help='path to dataset root folder')
  parser.add_argument('--seg_root', default='./data/cervix_roi/', help='path to segmented dataset root folder')
  parser.add_argument('--mixture', action='store_true', help='use the mixture mode')

  parser.add_argument('--test_idx',  default='./data_utils/test.csv', help='path to test idx file, does not contain label')
  parser.add_argument('--rel_path',  action='store_true', help='show relative path instead of img name in output csv')

  # parser.add_argument('--batch_size', type=int, default=64,  help='input batch size')
  parser.add_argument('--workers',    type=int, default=4,   help='number of data loading workers')

  parser.add_argument('--model_file', default='./ckpt_b32_w8/model_final.pth',required=True, help='the checkpoint file of the model to submit')
  parser.add_argument('--arch', default='inception_v3', required=True, help='the name of nn architecture e.g. resnet18, vgg19 etc.')
  parser.add_argument('--extra_name', default='', help='extra name for the filename')

  parser.add_argument('--save_dir', default='./submission' )
  parser.add_argument('--ten_crop', action='store_true', help='use ten_crop for test')
  parser.add_argument('--forty_crop', action='store_true', help='use 40_crop for test, 4 scales * 10 crops')
  args = parser.parse_args()

  if 'val' in args.test_idx:
    args.rel_path = True
    args.save_dir = './val_submissions'
  # manual seed
  args.manual_seed = random.randint(0, 10000) # fix seed
  print("Random Seed: ", args.manual_seed)
  random.seed(args.manual_seed)
  np.random.seed(args.manual_seed)
  torch.manual_seed(args.manual_seed)

  args.cuda = torch.cuda.is_available()
  if args.cuda:
    print('using cuda')
    if cudnn.enabled:
      cudnn.benchmark = True
      print('using cudnn {}'.format(cudnn.version()))


  print(args)


  # Data augmentation and normalization for training
  # Just normalization for validation
  (scale_size, crop_size) = (370, 299) if 'inception_v3' in args.arch else (256, 224)

  # create the dataloader for test image
  idx_files =  args.test_idx

  if args.ten_crop: # TODO: test mix mode
    dsets = get_augmented_test_set(data_root=args.data_root, idx_file=idx_files,
                        scale_size=scale_size, crop_size=crop_size,
                        aug_type='ten_crop',
                        seg_root=args.seg_root, mixture=args.mixture)
  elif args.forty_crop:
    if 'inception_v3' in args.arch:
      scale_sizes = (340, 370, 400, 430)
    else:
      scale_sizes = (256, 288, 320, 352) # follow googlenet
    dsets = get_multi_scale_crop_test_set(data_root=args.data_root, idx_file=idx_files,
                        scale_sizes=scale_sizes, crop_size=crop_size,
                        aug_type='forty_crop',
                        seg_root=args.seg_root, mixture=args.mixture)
  else:
    data_transform = transforms.Compose([
          transforms.Scale(scale_size),
          transforms.CenterCrop(crop_size),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dsets = [MyImageFolder(root = args.data_root,
                      idx_file = idx_files,
                      transform = data_transform)]
    if args.mixture:
      seg_data_transform = transforms.Compose([
                  transforms.Scale(crop_size),
                  transforms.CenterCrop(crop_size),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      dsets = [MyImageFolder(root = args.data_root,
                          idx_file = idx_files,
                          transform = data_transform,
                          seg_transform = seg_data_transform,
                          seg_root = args.seg_root)
              ]

  dset_size = len(dsets[0])
  num_classes = 3

  model = nets.get_net(args, num_classes, pretrained=False)

  if args.cuda:
    model = model.cuda()

  # load the model from checkpoint
  state = torch.load(args.model_file)
  model.load_state_dict(state['state_dict'])
  model.eval()

  # prepare the file path to save results
  try:
    os.makedirs(args.save_dir)
  except OSError:
    pass


  # evaluate the model on test set
  softmax = nn.Softmax()

  def test_for_one_dset(dset):
    probs = np.zeros((dset_size, 3))
    names = []
    for i in tqdm(range(dset_size)):
      if args.mixture:
        input_data, input_seg, label, name = dset[i]
        mix_w = 0.5
      elif args.rel_path:
        input_data, label, name = dset.__getitem__(i, ret_path=True)
      else:
        input_data, label, name = dset[i]
      names.append(name)
      input_data.unsqueeze_(0)
      if args.mixture:
        input_seg.unsqueeze_(0)

      if args.cuda:
        input_data = Variable(input_data.cuda())
        if args.mixture:
          input_seg = Variable(input_seg.cuda())
      else:
        input_data = Variable(input_data)
        if args.mixture:
          input_seg = Variable(input_seg)


      output_data = model(input_data)
      pr = softmax(output_data)

      if args.mixture:
        output_seg = model(input_seg)
        pr = softmax(output_data*(1-mix_w) + output_seg*mix_w)

      result = pr.cpu().data.numpy()[0]

      probs[i, :] = result

      # free graph memory
      del output_data, pr
      if args.mixture:
        del output_seg
    return probs, names

  probs_list = []
  for i in tqdm(range(len(dsets))):
    if i == 0:
      probs, names = test_for_one_dset(dsets[i])
    else:
      probs, _ = test_for_one_dset(dsets[i])
    probs_list.append(probs)

  probs_final = sum(probs_list)/len(probs_list)

  filename = '{0}_{1}_{2}_{3}.csv'.format(args.arch,
                  os.path.basename(args.model_file).split('.')[0],
                  args.extra_name,
                  datetime.now().strftime('_%Y_%m_%d_%H_%M_%S'))
  header_ = ','.join(['image_name', 'Type_1', 'Type_2', 'Type_3'])
  with open(os.path.join(args.save_dir, filename), 'w') as f_csv:
    f_csv.write(header_+'\n')
    for i in range(dset_size):
      f_csv.write('{0},{1:.8f},{2:.8f},{3:.8f}\n'.format(names[i],
                probs_final[i,0], probs_final[i,1], probs_final[i,2]))




