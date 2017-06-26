#!/usr/bin/python
# -*- coding: utf-8 -*-
''' python 3.4+
author: Gu Wang, Yu Yang, Shi Yan, Junjie Wu
'''
from __future__ import division, absolute_import, print_function

import os, sys
import datetime
import shutil
import copy
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

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
from torch.utils.data import DataLoader, sampler

import torchsample as ts

from tensorboard_logger import configure, log_value

from data_utils.my_folder import MyImageFolder
import utils
from utils import * # KaggleLogLoss, BCE_Loss weights_init, optim_scheduler_ft
from lr_scheduler import ReduceLROnPlateau
from models import nets
from models import myresnet
from train_eval_funcs import *

parser = argparse.ArgumentParser()
# ./data/IntelMobileODT_Cervical_Cancer_Screening/
parser.add_argument('--data_root', default='./data/resized_data/', help='path to dataset root folder')
parser.add_argument('--seg_root', default='./data/cervix_roi/', help='path to segmented dataset root folder')
parser.add_argument('--mixture', action='store_true', help='use the mixture mode')
parser.add_argument('--train_idx', default='./data_utils/train_idx_0505.csv',
                    help='path to train idx file, set to ./data_utils/val_idx.csv to do quick debug the train process')
parser.add_argument('--val_idx',   default='./data_utils/val_idx_0505.csv', help='path to val idx file')
parser.add_argument('--test_idx',  default='./data_utils/test_stg1.csv', help='path to test idx file, does not contain label')

parser.add_argument('--arch',   default='resnet18',
                    choices=['resnet18', 'resnet34', 'myresnet18', 'myresnet34', 'resnet50', 'resnet101',
                             'myresnet101', 'resnet152', 'myresnet152', 'vgg16', 'vgg19', 'inception_v3',
                             'inception_v3_dropout', 'inception_v3_alldrop'],
                    help='network architecture')
parser.add_argument('--log_dir',   default='./log',  help='log dir')
parser.add_argument('--exp_name',  default='exp_name',  help='experiment name')
parser.add_argument('--ckpt_dir',  default='./ckpt', help='checkpoint dir')

parser.add_argument('--batch_size', type=int, default=64,  help='input batch size')
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--workers',    type=int, default=4,   help='number of data loading workers')
parser.add_argument('--pin_memory', type=int, default=1,   help='set to 0 if system freezes.')

parser.add_argument('--ckpt_epoch',  default=1, type=int, help='How often to do checkpoint')
parser.add_argument('--load_ckpt', default='./ckpt/model_epoch_25.pth', help='the file path to the ckpt to be restored')
parser.add_argument('--resume', action='store_true', help='whether to resume training from load_ckpt')
parser.add_argument('--pretrained', default=1, type=int, help='use pre-trained model')

parser.add_argument('--manual_seed', type=int, default=-1, help='random seed. Set -1 to do random seed in [0,10000].')
parser.add_argument('--n_epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--optimizer', default='nag', help='optimizer to use: rmsprop | adam | adadelta | sgd | nag (Nesterov Accelerated Gradient)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--warmup', default=False, type=bool,
                    help='warmup training')
parser.add_argument('--warm_lr', default=1e-4, type=float,
                    help='warmup learning rate')
parser.add_argument('--warm_epochs', default=5, type=int,
                    help='warmup epochs')
parser.add_argument('--warmup_type', default='constant', choices=['constant', 'linear'],
                    help='warmup type')
parser.add_argument('--lr_decay_factor', default=0.1, type=float,
                    help='learning rate decay factor')
parser.add_argument('--lr_decay_epoch', default=10, type=int,
                    help='how many epochs to decay learning rate')
parser.add_argument('--cos_lr', default=0, type=int,
                    help='whether to use cosine lr schedule')
parser.add_argument('--M', default=10, type=int,
                    help='how many models the cos_schedule will save')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--weighted_sample', action='store_true', help='use WeightedRandomSampler to load data')
# weighted loss option
parser.add_argument('--weighted_loss', type=int, default=0,  help='weighted loss, default 0 means no weighted loss')
parser.add_argument('--slow_base', action='store_true', help='slow base params while finetuning')

# data augmentation
parser.add_argument('--rotate', type=int, default=0,
                  help='random rotation in [-degree, degree] for training set, default is 0, set 10 for example')
parser.add_argument('--rotate_clockwise', type=int, default=0,
                  help='whether to randomly rotate in [0, 90, 180, 270] for training set. Default is 0, no rotation')
parser.add_argument('--zoom', action='store_true',
                  help='random zoom in [0.8, 1.1] for training set, default is false')
parser.add_argument('--color_aug', action='store_true',
                  help='random color augmentations, including brightness, contrast, saturation, gamma(lighting?), for training set, default is false')
parser.add_argument('--loss', default='CE_Loss', choices=['CE_Loss', 'BCE_Loss'], help='loss to optimize')

args = parser.parse_args()


# manual seed
if args.manual_seed == -1:
  args.manual_seed = random.randint(0, 10000) # fix seed
print("Random Seed: ", args.manual_seed)
random.seed(args.manual_seed)
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)

try:
  os.makedirs(args.ckpt_dir)
except OSError:
  pass

args.cuda = torch.cuda.is_available()
if args.cuda:
  print('using cuda')
  # torch.cuda.manual_seed(args.manual_seed) # seed for current gpu
  if cudnn.enabled:
    cudnn.benchmark = True
    print('using cudnn {}'.format(cudnn.version()))

print(args)


# Data augmentation and normalization for training
(scale_size, crop_size) = (370, 299) if 'inception_v3' in args.arch else (256, 224)


extra_train_augments = [] # perform on Tensor
if args.rotate > 0:
  extra_train_augments.append(ts.transforms.Rotate(args.rotate))
if args.rotate_clockwise:
  extra_train_augments.append(RandomDiscreteRotate([0, 90, 180, 270]))
if args.zoom:
  extra_train_augments.append(ts.transforms.Zoom([0.7, 1.1]))
if args.color_aug:
  extra_train_augments += [ts.transforms.RandomBrightness(-0.1, 0.1),
                           ts.transforms.RandomContrast(0.7, 1.3),
                           ts.transforms.RandomSaturation(-0.2, 0.2),
                           ts.transforms.RandomGamma(0.6, 1.4)]

data_transforms = {
    'train': transforms.Compose([
        # transforms.Scale(scale_size), # the smaller edge
        transforms.RandomCrop(crop_size),
        # transforms.RandomSizedCrop(crop_size), # no need to scale before it
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor() ]
        + extra_train_augments
        + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    # val and test
    'val': transforms.Compose([
        transforms.Scale(scale_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(scale_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }

seg_data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(crop_size), # the smaller edge
        transforms.RandomCrop(crop_size),
        transforms.RandomSizedCrop(crop_size), # no need to scale before it
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor() ]
        + extra_train_augments
        + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    # val and test
    'val': transforms.Compose([
        transforms.Scale(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }


# tree -d data/IntelMobileODT_Cervical_Cancer_Screening
# data/IntelMobileODT_Cervical_Cancer_Screening
# ├── additional
# │   ├── Type_1
# │   ├── Type_2
# │   └── Type_3
# ├── test
# │   └── test
# └── train
#     ├── Type_1
#     ├── Type_2
#     └── Type_3
data_splits = ['train', 'val', 'test']

idx_files = {'train': args.train_idx,
             'val'  : args.val_idx,
             'test' : args.test_idx}

dsets = {}
if args.mixture:
  dsets = {x: MyImageFolder(root = args.data_root,
                            idx_file = idx_files[x],
                            transform = data_transforms[x],
                            seg_transform = seg_data_transforms[x],
                            seg_root = args.seg_root)
              for x in data_splits}
else:
  dsets = {x: MyImageFolder(root = args.data_root,
                            idx_file = idx_files[x],
                            transform = data_transforms[x])
              for x in data_splits}

shuffle_options = {'train': True, 'val': False, 'test': False}

if args.weighted_sample:
  sample_weight = dsets['train'].get_sample_weights()
  samplers = {'train': sampler.WeightedRandomSampler(weights=sample_weight, num_samples=len(dsets['train'])),
            'val'  : None,
            'test' : None}
else: # RandomSampler in train phase, SequentialSampler in val and test phase.
  samplers = {'train': None, 'val'  : None, 'test' : None}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
                                               shuffle=shuffle_options[x],
                                               sampler=samplers[x], #  If specified, the 'shuffle' argument is ignored
                                               num_workers=args.workers,
                                               pin_memory=bool(args.pin_memory))
                   for x in data_splits}

dset_sizes = {x: len(dsets[x]) for x in data_splits}
dset_classes = dsets['train'].classes
num_classes = len(dset_classes)


# get net
# =====================================================
model = nets.get_net(args, num_classes, pretrained=bool(args.pretrained))
if args.resume:
  epoch_trained, model, best_epoch, best_logloss, best_acc, best_model = resume_checkpoint(model=model, ckpt_path=args.load_ckpt)
if args.cuda:
  model = model.cuda()


# set loss
# =====================================================
weight = loss_weight(args, dset_loaders) if args.weighted_loss == 1 else None
if args.loss == 'CE_Loss':
  criterion = nn.CrossEntropyLoss(weight=weight)
elif args.loss == 'BCE_Loss':
  criterion = BCE_Loss(weight=weight) # -1/N sum( y*log(y_pred) + (1-y)*log(1-y_pred))
else:
  print('No such {}'.format(args.loss))
log_loss = KaggleLogLoss()

if args.cuda:
  criterion = criterion.cuda()
  log_loss = log_loss.cuda()

softmax = nn.Softmax()


# tensorboard_logger
# ===================================================================
if args.exp_name == 'exp_name':
  configure(os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
else:
  configure(os.path.join(args.log_dir, args.exp_name))

if args.resume:
  model = train_model(args, model, softmax, criterion, log_loss, optim_scheduler_ft,
                      dset_loaders, dset_sizes, num_epochs=args.n_epoch, epoch_trained=epoch_trained,
                      best_epoch_=best_epoch, best_model_logloss_=best_logloss, best_model_acc_=best_acc, best_model_=best_model)
else:
  model = train_model(args, model, softmax, criterion, log_loss, optim_scheduler_ft,
                      dset_loaders, dset_sizes, num_epochs=args.n_epoch, epoch_trained=0)


# save final model, TODO: may need to add more info
# ==================================================================
save_checkpoint(state={#'epoch': epoch_trained+args.n_epoch,
                      'state_dict':model.state_dict()},
                save_path='{0}/model_final.pth'.format(args.ckpt_dir))

evaluate_model(args, model, softmax, criterion, log_loss, dset_loaders, dset_sizes)

