''' python 3.4+
author: Gu Wang
'''
from __future__ import division, absolute_import, print_function

import os
import sys
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
from models import myresnet
from models import my_inception
from utils import *

from models.stn_module import stn_module


def get_dcnn(arch, num_classes, pretrained):
  if arch == 'resnet18':
    model = models.resnet18(pretrained = pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if not pretrained:
      model.apply(weights_init)
  elif arch == 'myresnet18':
    model = myresnet.myresnet18(pretrained = pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if not pretrained:
      model.apply(weights_init)
  elif arch == 'resnet34':
    model = models.resnet34(pretrained = pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if not pretrained:
      model.apply(weights_init)
  elif arch == 'myresnet34':
    model = myresnet.myresnet34(pretrained = pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if not pretrained:
      model.apply(weights_init)
  elif arch == 'resnet50':
    model = models.resnet50(pretrained = pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if not pretrained:
      print('initializing the resnet50 ...')
      model.apply(weights_init)
  elif arch == 'resnet101':
    model = models.resnet101(pretrained = pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if not pretrained:
      model.apply(weights_init)
  elif arch == 'myresnet101':
    model = myresnet.myresnet101(pretrained = pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if not pretrained:
      model.apply(weights_init)
  elif arch == 'resnet152':
    model = models.resnet152(pretrained = pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if not pretrained:
      model.apply(weights_init)
  elif arch == 'myresnet152':
    model = myresnet.myresnet152(pretrained = pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if not pretrained:
      model.apply(weights_init)
  elif arch == 'vgg16':
    model = models.vgg16(pretrained = pretrained)
    mod = list(model.classifier.children())[:-1] + [torch.nn.Linear(4096, num_classes)]
    new_classifier = torch.nn.Sequential(*mod)
    model.classifier = new_classifier
  elif arch == 'vgg19':
    model = models.vgg19(pretrained = pretrained)
    mod = list(model.classifier.children())[:-1] + [torch.nn.Linear(4096, num_classes)]
    new_classifier = torch.nn.Sequential(*mod)
    model.classifier = new_classifier
    # model = nets.Vgg19(num_classes, pretrained=bool(args.pretrained), bn_after_act=False)
  elif arch == 'inception_v3':
    model = models.inception_v3(pretrained = pretrained,
                                transform_input=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
  elif arch == 'inception_v3_dropout':
    model = models.inception_v3(pretrained = pretrained,
                                transform_input=True)
    num_ftrs = model.fc.in_features # 2048

    model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            # True will cause: "one of the variables needed for gradient computation
            #   has been modified by an inplace operation"
            nn.Linear(num_ftrs, num_classes),
          )

    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
  elif arch == 'inception_v3_alldrop':
    model = my_inception.my_inception_v3(pretrained = pretrained,
                                transform_input=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
  else:
    print('No {}!'.format(arch))
  return model


class stn_dcnn(nn.Module):
  """docstring for stn_dcnn"""
  def __init__(self, arch, num_classes, dcnn_pretrained):
    super(stn_dcnn, self).__init__()
    self.dcnn_arch = arch[:-4] # assume that the arch is in form of 'xxxxdcnnarch_stn'
    dcnn_inputsize = 299 if 'inception_v3' in self.dcnn_arch else 224
    self.stn = stn_module(dcnn_inputsize)
    self.dcnn = get_dcnn(self.dcnn_arch, num_classes, dcnn_pretrained)

  def forward(self, x):
    x = x.transpose(1,2).transpose(2,3)
    if self.training:
      x, stn_aux = self.stn(x)
    else:
      x = self.stn(x)
    x= x.transpose(2,3).transpose(1,2)
    if 'inception_v3' in self.dcnn_arch and self.training:
      x, incep_aux = self.dcnn(x)
    else:
      x = self.dcnn(x)

    if 'inception_v3' in self.dcnn_arch and self.training:
      return x, stn_aux, incep_aux
    elif 'inception_v3' not in self.dcnn_arch and self.training:
      return x, stn_aux
    else:
      return x



def get_net(args, num_classes, pretrained):
  # initial the model
  if 'stn' in args.arch:
    model = stn_dcnn(args.arch, num_classes, pretrained)
  else:
    model = get_dcnn(args.arch, num_classes, pretrained)
  return model


class Vgg19(nn.Module):
  def __init__(self, num_classes, pretrained=False,
              bn_after_act=False, bn_before_act=False):
    super(Vgg19, self).__init__()

    self.pretrained = pretrained
    self.bn_before_act = bn_before_act
    self.bn_after_act = bn_after_act

    model = models.vgg19(pretrained = pretrained)
    self.features = model.features


    self.fc17 = nn.Linear(512 * 7 * 7, 4096)
    self.bn17 = nn.BatchNorm1d(4096)
    self.fc18 = nn.Linear(4096, 4096)
    self.bn18 = nn.BatchNorm1d(4096)
    self.fc19 = nn.Linear(4096, num_classes)

    self._initialize_weights()

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.fc17(x)
    if self.bn_before_act:
      x = F.relu(self.bn17(x))
    else: # bn after act
      x = self.bn17(F.relu(x))
    x = self.fc18(x)
    if self.bn_before_act:
      x = F.relu(self.bn18(x))
    else:
      x = self.bn18(F.relu(x))
    x = self.fc19(x)

    return x

  def _initialize_weights(self):
    if not self.pretrained:
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
          if m.bias is not None:
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
          n = m.weight.size(1)
          m.weight.data.normal_(0, 0.01)
          m.bias.data.zero_()
    else:
      for m in self.modules():
        if isinstance(m, nn.BatchNorm1d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
          n = m.weight.size(1)
          m.weight.data.normal_(0, 0.01)
          m.bias.data.zero_()


