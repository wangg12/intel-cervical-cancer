''' python 3.4+
author: Gu Wang
'''
from __future__ import division, absolute_import, print_function

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.init as init
import torchvision

import random
from torchsample.transforms import *
from data_utils.my_folder import MyImageFolder
from torchvision import transforms
from lr_scheduler import cosine_anneal_schedule, warmup_scheduler

def convert_to_one_hot(y, num_class):
  return torch.zeros(y.size()[0], num_class).scatter_(1, y.unsqueeze(1), 1.)


class KaggleLogLoss(nn.Module):
  """kaggle log loss function.
  -1/N sum(y*log(y_pred))
  the submitted probabilities are divided by row_sum,
      and then max(min(p, 1-1e-15), 1e-15), can be achieved by torch.clamp

  TODO: not sure the order of these two operations
  """
  def __init__(self):
    super(KaggleLogLoss, self).__init__()

  def forward(self, y_pred, y_true_one_hot):
    '''
    y_pred: [B,C],
    y_true_one_hot: [B,C], Variable torch.LongTensor
    '''
    ## y_pred has passed through softmax
    # do not average over batch size here
    loss = - torch.mean(torch.sum(y_true_one_hot * torch.log(torch.clamp(y_pred, min=1e-15, max=1-1e-15)), 1))

    # loss = - torch.mean(y_true_one_hot * F.log_softmax(y_pred)) # y_pred should not be passed through softmax
    # loss = - torch.mean(y_true_one_hot * torch.log(y_pred + 1e-15)) # y_pred has passed through softmax
    # print(loss.data[0])
    return loss


class BCE_Loss(nn.Module):
  """docstring for BCE_Loss"""
  def __init__(self, weight):
    super(BCE_Loss, self).__init__()
    self.loss = nn.BCELoss(weight=weight)

  def forward(self, input, target):
    # input: raw output of nn (without softmax)
    # target: Variable of raw labels (not ont-hot)
    target_tensor = target.data.cpu()
    labels_one_hot = convert_to_one_hot(target_tensor, num_class=3)
    labels_one_hot = Variable(labels_one_hot.cuda()) if input.is_cuda else Variable(labels_one_hot) # wrap it into variable

    softmax = nn.Softmax()

    return self.loss(softmax(input), labels_one_hot)


def optim_scheduler_ft(model, epoch, optimizer_name='rmsprop', slow_base=True,
                       init_lr=0.001, lr_decay_epoch=10, lr_decay_factor=0.9,
                       momentum=0.9, weight_decay=1e-4,
                       beta1=0.9,
                       warmup=False, warm_lr=1e-4, warm_epochs=5, warmup_type='constant',
                       cos_schedule=False, cos_schedule_params=None):
  '''exponentially decrease the learning rate once every few epochs
  beta1: beta1 for adam, default is 0.9
  cos_schedule_params: for example: {'T': 100, 'M': 10, 'init_lr': 0.1}
  ----
  optimizer: the re-scheduled optimizer

  '''
  if cos_schedule == False:
    if warmup:
      if epoch<=warm_epochs:
        lr = warmup_scheduler(epoch, warm_lr=warm_lr, warm_epochs=warm_epochs, warmup_type=warmup_type, target_lr=init_lr)
      else:
        lr = init_lr * (lr_decay_factor**((epoch - 1 - warm_epochs) // lr_decay_epoch))
        if (epoch - 1 - warm_epochs) % lr_decay_epoch == 0:
          print('learning rate is set to {}'.format(lr))
    else:
      lr = init_lr * (lr_decay_factor**((epoch-1) // lr_decay_epoch))

      if (epoch - 1) % lr_decay_epoch == 0:
        print('learning rate is set to {}'.format(lr))
  else: # cosine schedule
    if warmup:
      if epoch<=warm_epochs:
        lr = warmup_scheduler(epoch, warm_lr=warm_lr, warm_epochs=warm_epochs, warmup_type=warmup_type, target_lr=init_lr)
      else:
        lr = cosine_anneal_schedule(epoch-1-warm_epochs, **cos_schedule_params)
    else:
      lr = cosine_anneal_schedule(epoch-1, **cos_schedule_params)
  print('epoch:{}, lr:{}'.format(epoch, lr))
  if slow_base:
    if isinstance(model, torchvision.models.vgg.VGG):
      ignored_params = model.classifier.parameters()
    elif isinstance(model, torchvision.models.resnet.ResNet) or isinstance(model, torchvision.models.inception.Inception3):
      ignored_params = model.fc.parameters()
  else:
    ignored_params = list()
  ignored_params_id = list(map(id, ignored_params))
  base_params = filter(lambda p: id(p) not in ignored_params_id, model.parameters())

  optimizer_name = optimizer_name.lower()
  if optimizer_name == 'adam':
    if slow_base:
      optimizer = optim.Adam([
        {'params': base_params},
        {'params': ignored_params, 'lr': lr}
      ], lr=lr * 0.1, betas=(beta1, 0.999), eps=1e-08, weight_decay=0)
    else:
      optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999), eps=1e-08, weight_decay=0)
  elif optimizer_name == 'adadelta': # lr=1.0 is recommended?
    if slow_base:
      optimizer = optim.Adadelta([
        {'params': base_params},
        {'params': ignored_params, 'lr': lr}
      ], lr=lr * 0.1, rho=0.9, eps=1e-06, weight_decay=0)
    else:
      optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=0)
  elif optimizer_name == 'rmsprop':
    if slow_base:
      optimizer = optim.RMSprop([
        {'params': base_params},
        {'params': ignored_params, 'lr': lr}
      ], lr=lr * 0.1, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    else:
      optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
  elif optimizer_name == 'nag':
    if slow_base:
      optimizer = optim.SGD([
        {'params': base_params},
        {'params': ignored_params, 'lr': lr}
      ], lr=lr * 0.1, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    else:
      optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
  elif optimizer_name == 'sgd':
    if slow_base:
      optimizer = optim.SGD([
        {'params': base_params},
        {'params': ignored_params, 'lr': lr}
      ], lr=lr * 0.1, momentum=momentum, weight_decay=weight_decay)
    else:
      optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
  else:
    optimizer = None
    print("No optimizer: {}!".format(optimizer_name))
  return optimizer, lr


def weights_init(m):
  # classname = m.__class__.__name__
  if isinstance(m, nn.Conv2d):
    #print('init conv2d')
    #init.xavier_uniform(m.weight.data, gain=np.sqrt(2.0))
    init.kaiming_uniform(m.weight.data, mode='fan_in')
    # m.weight.data.normal_(0.0, 0.02)
  if isinstance(m, nn.Linear):
    #print('init fc')
    init.kaiming_uniform(m.weight.data, mode='fan_in')
    # size = m.weight.size()
    # fan_out = size[0] # number of rows
    # fan_in = size[1] # number of columns
    # variance = np.sqrt(2.0/(fan_in + fan_out))
    # m.weight.data.uniform_(0.0, variance)


def imshow_tensor(inp):
  """Imshow for Tensor.
  inp: (3,h,w), typically torch's image are stored in (c,h,w),
       because common Tensors in neural networks in torch are in BCHW format
  """
  inp = inp.numpy().transpose((1, 2, 0)) # (h,w,3)
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  plt.imshow(inp)


class RandomDiscreteRotate(object):

    def __init__(self,
                 rotation_range,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between degrees in the given list. If the image
        has multiple channels, the same rotation will be applied to each channel.
        Arguments
        ---------
        rotation_range : list
            image will be rotated between degrees given in list
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
        fill_value : float
            the value to fill the empty space with if fill_mode='constant'
        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        self.rotation_range = rotation_range
        if not isinstance(interp, (tuple,list)):
            interp = (interp, interp)
        self.interp = interp
        self.lazy = lazy

    def __call__(self, x, y=None):
        k = random.randint(0, len(self.rotation_range)-1)
        degree = self.rotation_range[k]

        if self.lazy:
            return Rotate(degree, lazy=True)(x)
        else:
            if y is None:
                x_transformed = Rotate(degree,
                                       interp=self.interp)(x)
                return x_transformed
            else:
                x_transformed, y_transformed = Rotate(degree,
                                                      interp=self.interp)(x,y)
                return x_transformed, y_transformed


def get_augmented_test_set(data_root, idx_file,
              scale_size, crop_size, aug_type='ten_crop',
              seg_root=None, mixture=False):
  dsets = []
  if aug_type == 'ten_crop':
    crop_types = [0, 1, 2, 3, 4]
    # 0: center crop,
    # 1: top left crop, 2: top right crop
    # 3: bottom right crop, 4: bottom left crop
    flips = [0, 1] # 0: no flip, 1: horizontal flip
    for i in crop_types:
      for j in flips:
        data_transform = transforms.Compose([
          transforms.Scale(scale_size),
          # transforms.CenterCrop(crop_size),
          transforms.ToTensor(),
          RandomFlip(flips[j]),
          SpecialCrop((crop_size, crop_size), crop_type=crop_types[i]),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if mixture:
          seg_transform = transforms.Compose([
              transforms.Scale(crop_size),
              # transforms.CenterCrop(crop_size),
              transforms.ToTensor(),
              RandomFlip(flips[j]),
              # SpecialCrop(crop_size=(crop_size, crop_size), crop_type=crop_types[i]),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])

          dsets.append(MyImageFolder(root = data_root,
                      idx_file = idx_file,
                      transform = data_transform,
                      seg_transform = seg_transform,
                      seg_root = seg_root))
        else:
          dsets.append(MyImageFolder(root = data_root,
                      idx_file = idx_file,
                      transform = data_transform))

  return dsets


def get_multi_scale_crop_test_set(data_root, idx_file,
              scale_sizes, crop_size, aug_type='forty_crop',
              seg_root=None, mixture=False):
  dsets = []
  if aug_type == 'forty_crop':
    for scale_size in scale_sizes:
      crop_types = [0, 1, 2, 3, 4]
      # 0: center crop,
      # 1: top left crop, 2: top right crop
      # 3: bottom right crop, 4: bottom left crop
      flips = [0, 1] # 0: no flip, 1: horizontal flip
      for i in crop_types:
        for j in flips:
          data_transform = transforms.Compose([
            transforms.Scale(scale_size),
            # transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            RandomFlip(flips[j]),
            SpecialCrop((crop_size, crop_size), crop_type=crop_types[i]),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])
          if mixture:
            seg_transform = transforms.Compose([
                transforms.Scale(crop_size),
                # transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                RandomFlip(flips[j]),
                # SpecialCrop(crop_size=(crop_size, crop_size), crop_type=crop_types[i]),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            dsets.append(MyImageFolder(root = data_root,
                        idx_file = idx_file,
                        transform = data_transform,
                        seg_transform = seg_transform,
                        seg_root = seg_root))
          else:
            dsets.append(MyImageFolder(root = data_root,
                        idx_file = idx_file,
                        transform = data_transform))

  return dsets


def loss_weight(args, dset_loaders):
  '''
  get the weights of losses of different types

  '''
  total_type1 = 0
  total_type2 = 0
  total_type3 = 0

  for data in dset_loaders['train']:
    inputs, labels, _ = data
    labels_one_hot = convert_to_one_hot(labels,num_class=3)

    if args.cuda:
      _, labels = Variable(inputs.cuda()), Variable(labels.cuda())
      labels_one_hot = Variable(labels_one_hot.cuda())
    else:
      _, labels = Variable(inputs), Variable(labels)
      labels_one_hot = Variable(labels_one_hot)

    total_type1 += torch.sum(labels.data == 0)
    total_type2 += torch.sum(labels.data == 1)
    total_type3 += torch.sum(labels.data == 2)

  weight_type1 = 1/total_type1
  weight_type2 = 1/total_type2
  weight_type3 = 1/total_type3

  total = weight_type1 + weight_type2 + weight_type3

  weight = torch.FloatTensor([weight_type1/total,weight_type2/total,weight_type3/total])

  return weight



"""
def visualize_model(model, num_images=5):
  for i, data in enumerate(dset_loaders['val']):
    inputs, labels, _ = data
    if args.cuda:
      inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
      inputs, labels = Variable(inputs), Variable(labels)


    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)

    plt.figure()
    imshow(inputs.cpu().data[0])
    plt.title('pred: {}'.format(dset_classes[labels.data[0]]))
    plt.show()

    if i == num_images - 1:
      break
"""
