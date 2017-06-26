''' python 3.4+
author: Yu YANG
'''
from __future__ import division, absolute_import, print_function

import torch
import math
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(THIS_DIR, '../external/stn.pytorch/script/'))
from modules.stn import STN
from modules.gridgen import AffineGridGen


class localization_net(nn.Module):
  """docstring for localization_net"""
  def __init__(self):
    super(localization_net, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False)
    self.bn2 = nn.BatchNorm2d(64)
    self.fc1 = nn.Linear(28*28*64, 512)
    self.fc2 = nn.Linear(512, 6)
    self.relu = nn.ReLU(inplace=True)

    # initialize the parameter
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    # initialize the final fc with identity matrix
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    fc2Params = list(self.fc2.parameters())
    fc2Params[1].data = torch.from_numpy(initial)


  def forward(self, x):
    x= x.transpose(2,3).transpose(1,2)
    # print('input size to loc net {}'.format(x.size()))
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = x.view(x.size(0),-1)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)

    return x


class stn_module(nn.Module):
  """docstring for stn_module"""
  def __init__(self, dcnn_inputsize):
    super(stn_module, self).__init__()
    self.loc_net = localization_net()
    self.sampler = STN()
    self.grid_generator = AffineGridGen(dcnn_inputsize, dcnn_inputsize, lr=0.001, aux_loss=True)


  def forward(self, x):
    transformer_para = self.loc_net(x)
    transformer_para = transformer_para.view(transformer_para.size(0),2,3)
    # print('input size to grid gen {}'.format(transformer_para.size()))
    grid, aux = self.grid_generator(transformer_para)

    out = self.sampler(x, grid)

    if self.training:
      return out, aux
    return out
