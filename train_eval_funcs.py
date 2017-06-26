'''
author: Gu Wang
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
from utils import * # KaggleLogLoss, weights_init, optim_scheduler_ft
from lr_scheduler import ReduceLROnPlateau
from models import nets
from models import myresnet



def train_model(args, model, softmax, criterion, log_loss, optim_scheduler,
                dset_loaders, dset_sizes,
                num_epochs, epoch_trained=0,
                best_epoch_=-1, best_model_logloss_=None, best_model_acc_=0.0, best_model_=None):
  '''
  optim_scheduler: a function which returns an optimizer object when called as optim_scheduler(model, epoch)
      This is useful when we want to change the learning rate or restrict the parameters we want to optimize.
  '''
  since = time.time()

  best_model = best_model_
  best_model_acc = best_model_acc_
  best_model_logloss = best_model_logloss_
  best_epoch = best_epoch_

  n_batches = {'train': len(dset_loaders['train']),
                'val': len(dset_loaders['val']),
                'test': len(dset_loaders['test'])}

  for epoch in range(epoch_trained+1, epoch_trained+1+num_epochs):
    print('Epoch {}/{}'.format(epoch, epoch_trained+num_epochs))
    print('-' * 10)

    val_acc = 0.0
    val_logloss = None
    # Each epoch has a training and validation phase
    for phase in ['train', 'val', 'test']:
      if phase == 'train':
        model.train()
        optimizer, lr = optim_scheduler(model, epoch, optimizer_name=args.optimizer, init_lr=args.lr,
                                    slow_base=args.slow_base,
                                    lr_decay_factor=args.lr_decay_factor,
                                    lr_decay_epoch=args.lr_decay_epoch,
                                    momentum=args.momentum, weight_decay=args.weight_decay,
                                    warmup=args.warmup, warm_lr=args.warm_lr, warm_epochs=args.warm_epochs, warmup_type = args.warmup_type,
                                    cos_schedule=bool(args.cos_lr), cos_schedule_params={'T': num_epochs, 'M': args.M, 'init_lr': args.lr})
        log_value('lr', lr, step = epoch)
      elif phase == 'val' or phase == 'test':
        model.eval()

      running_loss = 0.0
      running_log_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      step = 1
      for data in tqdm(dset_loaders[phase]):
        # get the inputs
        if args.mixture:
          inputs, inputs_seg, labels, _ = data
          mix_w = 0.5
        else:
          inputs, labels, _ = data
        labels_one_hot = utils.convert_to_one_hot(labels, num_class=3)

        # wrap them in Variable
        if args.cuda:
          inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
          labels_one_hot = Variable(labels_one_hot.cuda())
        else:
          inputs, labels = Variable(inputs), Variable(labels)
          labels_one_hot = Variable(labels_one_hot)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward =====================================================================
        if 'inception_v3' in args.arch and 'stn' in args.arch and phase == 'train':
          outputs, stn_aux, incep_aux = model(inputs)
        elif 'inception_v3' in args.arch and 'stn' not in args.arch and phase == 'train':
          outputs, incep_aux = model(inputs)
        elif 'inception_v3' not in args.arch and 'stn' in args.arch and phase == 'train':
          outputs, stn_aux = model(inputs)
        else:
          outputs = model(inputs)

        # mixture mode
        # CATION: stn with mixture mode is incompleted
        if args.mixture:
          inputs_seg = Variable(inputs_seg.cuda()) if args.cuda else Variable(inputs_seg)
          if 'inception_v3' in args.arch and phase == 'train':
            outputs_seg, aux_outputs_seg = model(inputs_seg)
          else:
            outputs_seg = model(inputs_seg)

        if args.mixture:
          # CATION: stn with mixture mode is incompleted
          _, preds = torch.max(outputs.data*(1-mix_w) + outputs_seg.data*mix_w, dim=1)
          loss = criterion(outputs*(1-mix_w) + outputs_seg*mix_w, labels)
          if 'inception_v3' in args.arch and phase == 'train':
            incep_auxloss = criterion(incep_aux*(1-mix_w) + aux_outputs_seg*mix_w, labels)
          loss_log = log_loss(softmax(outputs*(1-mix_w) + outputs_seg*mix_w), labels_one_hot)
        else:
          _, preds = torch.max(outputs.data, dim=1)
          loss = criterion(outputs, labels)
          if 'inception_v3' in args.arch and phase == 'train':
            incep_auxloss = criterion(incep_aux, labels)
          loss_log = log_loss(softmax(outputs), labels_one_hot)

        # backward + optimize only if in training phase =====================================
        if phase == 'train':
          if 'inception_v3' in args.arch and 'stn' in args.arch:
            stn_auxloss = torch.mean(stn_aux)
            total_loss = loss + 0.1*stn_auxloss + incep_auxloss
            total_loss.backward()
          elif 'inception_v3' not in args.arch and 'stn' in args.arch:
            stn_auxloss = torch.mean(stn_aux)
            total_loss = loss + 0.1*stn_auxloss
            total_loss.backward()
          elif 'inception_v3' in args.arch and 'stn' not in args.arch:
            total_loss = loss + incep_auxloss
            total_loss.backward()
          else:
            loss.backward()
          optimizer.step()

        # statistics
        running_loss += loss.data[0]*inputs.size()[0]
        running_log_loss += loss_log.data[0]*inputs.size()[0]
        running_corrects += torch.sum(preds == labels.data)

        global_step = (epoch-1)*n_batches[phase] + step


        log_value(phase+'_{}'.format(args.loss), loss.data[0], step = global_step)
        log_value(phase+'_log_loss', loss_log.data[0], step = global_step)
        log_value(phase+'_acc',
                  torch.mean((preds == labels.data).type_as(torch.FloatTensor())),
                  step = global_step)
        step += 1

        # free the graph to avoid memory increase
        del outputs, loss, loss_log
        if args.mixture:
          del outputs_seg

      epoch_loss = running_loss / dset_sizes[phase]
      epoch_log_loss = running_log_loss / dset_sizes[phase]
      epoch_acc = running_corrects / dset_sizes[phase]

      print('{} {}_Loss: {:.4f}, Log_loss: {:.4f}, Acc: {:.4f}'.format(
                phase, loss_name, epoch_loss, epoch_log_loss, epoch_acc))
      log_value('epoch{}_{}'.format(phase, args.loss), epoch_loss, step=epoch)
      log_value('epoch{}_log_loss'.format(phase), epoch_log_loss, step=epoch)
      log_value('epoch{}_acc'.format(phase), epoch_acc, step=epoch)

      if phase == 'val':
        val_acc = epoch_acc
        val_logloss = epoch_log_loss
      # deep copy the model
      if phase == 'val' and (best_model_logloss is None or epoch_log_loss < best_model_logloss):
        best_model_acc = epoch_acc
        best_model_logloss = epoch_log_loss
        best_epoch = epoch
        best_model = copy.deepcopy(model)

    # do checkpointing
    if epoch % args.ckpt_epoch == 0:
      save_checkpoint(state={'epoch': epoch,
                             'val_logloss':val_logloss,
                             'val_acc':val_acc,
                             'state_dict':model.state_dict()},
              save_path='{0}/model_epoch_{1}.pth'.format(args.ckpt_dir, epoch))
      save_checkpoint(state={'epoch': best_epoch,
                             'val_logloss':best_model_logloss,
                             'val_acc':best_model_acc,
                             'state_dict':best_model.state_dict()},
              save_path='{0}/best_model.pth'.format(args.ckpt_dir))
    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
                  time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}, best epoch: {}'.format(best_model_acc, best_epoch))
  return best_model


def evaluate_model(args, model, softmax, criterion, log_loss,
                  dset_loaders, dset_sizes):
  '''
  evaluate per-class accuracy
  '''
  running_loss = 0.0
  running_log_loss = 0.0
  total_corrects = 0

  total_type1 = 0
  corrects_type1 = 0
  total_type2 = 0
  corrects_type2 = 0
  total_type3 = 0
  corrects_type3 = 0

  # switch to evaluate mode
  model.eval()

  for data in dset_loaders['val']:
    # get the inputs
    if args.mixture:
      inputs, inputs_seg, labels, _ = data
    else:
      inputs, labels, _ = data
    labels_one_hot = utils.convert_to_one_hot(labels, num_class=3)

    # wrap them in Variable
    if args.cuda:
      inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
      labels_one_hot = Variable(labels_one_hot.cuda())
      if args.mixture:
        inputs_seg = Variable(inputs_seg.cuda())
    else:
      inputs, labels = Variable(inputs), Variable(labels)
      labels_one_hot = Variable(labels_one_hot)
      if args.mixture:
        inputs_seg = Variable(inputs_seg)

    # forward
    outputs = model(inputs)
    if args.mixture:
      outputs_seg = model(inputs_seg)

    if args.mixture:
      _, preds = torch.max((outputs.data+outputs_seg.data)/2, dim=1)
      if args.CE_Loss:
        loss = criterion((outputs+outputs_seg)/2, labels)
      else:
        loss = criterion(softmax((outputs+outputs_seg)/2), labels_one_hot)
        # print(loss_log.size(), loss_log.data[0])
      loss_log = log_loss(softmax((outputs+outputs_seg)/2), labels_one_hot)
    else:
      _, preds = torch.max(outputs.data, dim=1)
      if args.CE_Loss:
        loss = criterion(outputs, labels)
      else:
        loss = criterion(softmax(outputs), labels_one_hot)
        # print(loss_log.size(), loss_log.data[0])
      loss_log = log_loss(softmax(outputs), labels_one_hot)

    # statistics
    running_loss += loss.data[0]*inputs.size()[0]
    running_log_loss += loss_log.data[0]*inputs.size()[0]
    total_corrects += torch.sum(preds == labels.data)

    total_type1 += torch.sum(labels.data == 0)
    corrects_type1 += torch.sum( (preds==labels.data)*(labels.data==0) )
    total_type2 += torch.sum(labels.data == 1)
    corrects_type2 += torch.sum( (preds==labels.data)*(labels.data==1) )
    total_type3 += torch.sum(labels.data == 2)
    corrects_type3 += torch.sum( (preds==labels.data)*(labels.data==2) )

  evaluate_loss = running_loss / dset_sizes['val']
  evaluate_log_loss = running_log_loss / dset_sizes['val']
  evaluate_total_acc = total_corrects / dset_sizes['val']
  evaluate_acc_type1 = corrects_type1 / total_type1
  evaluate_acc_type2 = corrects_type2 / total_type2
  evaluate_acc_type3 = corrects_type3 / total_type3

  print('Evaluation results')
  print('-' * 10)
  if args.CE_Loss:
    print('CE_Loss: {:.4f}, Log_loss: {:.4f}, Acc: {:.4f}'.format(
                  evaluate_loss, evaluate_log_loss, evaluate_total_acc))
  else:
    print('BCE_Loss: {:.4f}, Log_loss: {:.4f}, Acc: {:.4f}'.format(
                  evaluate_loss, evaluate_log_loss, evaluate_total_acc))
  print('Type1 acc: {:.4f}'.format(evaluate_acc_type1))
  print('Type2 acc: {:.4f}'.format(evaluate_acc_type2))
  print('Type3 acc: {:.4f}'.format(evaluate_acc_type3))


def save_checkpoint(state, save_path='checkpoint.pth'):
  '''
  state: dict, {'epoch': epoch,
               'val_logloss':val_logloss,
               'val_acc':val_acc,
               'state_dict':model.state_dict()}
  '''
  save_dir = os.path.split(save_path)[0]
  if os.path.isdir(save_dir) and (not os.path.exists(save_dir)):
    os.makedirs(save_dir)
  torch.save(state, save_path)
  # if is_best:
  #   shutil.copyfile(save_path, os.path.join(save_dir ,'model_best.pth'))


def resume_checkpoint(model, ckpt_path='checkpoint.pth'):
  '''
  state: dict, {'epoch': epoch,
               'val_logloss':val_logloss,
               'val_acc':val_acc,
               'state_dict':model.state_dict()}
  '''
  state = torch.load(ckpt_path)
  best_model = copy.deepcopy(model)
  model.load_state_dict(state['state_dict'])
  epoch = state['epoch']

  # load the best modle
  best_path = os.path.join(os.path.dirname(ckpt_path),'best_model.pth')
  if os.path.isfile(best_path):
    state = torch.load(best_path)
    best_model.load_state_dict(state['state_dict'])
    best_epoch = state['epoch']
    best_model_acc = state['val_acc']
    best_model_logloss = state['val_logloss']
  else:
    print('checkpoint file of best model does not exist!')

  return epoch, model, best_epoch, best_model_logloss, best_model_acc, best_model
