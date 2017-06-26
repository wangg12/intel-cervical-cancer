# -*- coding: utf-8 -*-
''' python 3.4+
author: Yu Yang, Gu Wang, Shi Yan
'''
from __future__ import division, absolute_import, print_function
import os, sys
import shutil
import copy
import time
import random
import argparse
import numpy as np
import csv
from tqdm import tqdm
from datetime import datetime

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# to solve the default PIL loader problem in torchvision

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from data_utils.my_folder import MyImageFolder
import utils
from utils import KaggleLogLoss
from utils import weights_init
from lr_scheduler import ReduceLROnPlateau
from models import nets

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_root', default='./data/resized_data/', help='path to dataset root folder')
  parser.add_argument('--val_idx',  default='./data_utils/val_idx_0505.csv',      help='path to test idx file, does not contain label')

  # parser.add_argument('--batch_size', type=int, default=64,  help='input batch size')
  parser.add_argument('--workers',    type=int, default=4,   help='number of data loading workers')

  parser.add_argument('--model_file', default='./ckpt_b32_w8/model_final.pth',required=True, help='the checkpoint file of the model to submit')
  parser.add_argument('--arch', default='resnet18', required=True, help='the name of nn architecture e.g. resnet18, vgg19 etc.')
  parser.add_argument('--save_dir', default='./submission' )
  args = parser.parse_args()

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
  (scale_size, crop_size) = (342, 299) if args.arch == 'inception_v3' else (256, 224)

  data_transform = transforms.Compose([
          transforms.Scale(scale_size),
          transforms.CenterCrop(crop_size),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

  # create the dataloader for test image
  idx_files =  args.val_idx

  dset = MyImageFolder(root = args.data_root,
                            idx_file = idx_files,
                            transform = data_transform)

  # dset_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size,
  #                                                shuffle=False,
  #                                                num_workers=args.workers,
  #                                                pin_memory=True)
  dset_size = len(dset)
  num_classes = 3

  # create the model
  if args.arch == 'resnet18':
    model = models.resnet18(pretrained = False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
  elif args.arch == 'resnet34':
    model = models.resnet34(pretrained = False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
  elif args.arch == 'resnet50':
    model = models.resnet50(pretrained = False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
  elif args.arch == 'resnet101':
    model = models.resnet101(pretrained = False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
  elif args.arch == 'resnet152':
    model = models.resnet152(pretrained = False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
  elif args.arch == 'vgg16':
    model = models.vgg16(pretrained = False)
    mod = list(model.classifier.children())[:-1] + [torch.nn.Linear(4096, num_classes)]
    new_classifier = torch.nn.Sequential(*mod)
    model.classifier = new_classifier
  elif args.arch == 'vgg19':
    model = models.vgg19(pretrained = False)
    mod = list(model.classifier.children())[:-1] + [torch.nn.Linear(4096, num_classes)]
    new_classifier = torch.nn.Sequential(*mod)
    model.classifier = new_classifier
    # model = nets.Vgg19(num_classes, pretrained=bool(args.pretrained), bn_after_act=False)
  elif args.arch == 'inception_v3':
    model = models.inception_v3(pretrained = False, transform_input=True, aux_logits=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
  else:
    print('No {}!'.format(args.arch))

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
  filename = args.arch + os.path.basename(args.model_file).split('.')[0] + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')+'.csv'

  # evaluate the model on test set
  softmax = nn.Softmax()

  with open(os.path.join(args.save_dir,filename), 'w') as csvfile:
    fieldnames = ['image_name', 'Type_1', 'Type_2', 'Type_3']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in tqdm(range(dset_size)):
      input_data, label, name = dset[i]
      input_data.unsqueeze_(0)

      if args.cuda:
        input_data = Variable(input_data.cuda())
      else:
        input_data = Variable(input_data)

      output_data = model(input_data)
      pr = softmax(output_data)
      result = pr.cpu().data.numpy()[0]
      # result = result.clip(0.005, 0.995)

      writer.writerow({'image_name': name,
                       'Type_1': result[0],
                      'Type_2': result[1],
                      'Type_3': result[2] })
