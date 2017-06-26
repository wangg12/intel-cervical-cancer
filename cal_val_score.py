from __future__ import print_function, division
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
import argparse
from utils import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--score_file',  default='./val_submissions/',required=True, help='path to val submission file')
  parser.add_argument('--val_idx',  default='./data_utils/val_idx_0505.csv', help='path to val idx file')
  args = parser.parse_args()
  
  f_score = open(args.score_file, 'r')
  i = 0
  scores = []
  for line in f_score:
    if i == 0:
      i += 1
      continue
    line_list = line.strip('\r\n').split(',')
    img_name, score_1, score_2, score_3 = line_list
    scores.append([float(score_1), float(score_2), float(score_3)])
    i += 1
  f_score.close()

  scores = np.array(scores).astype('float32')
  
  f_val = open(args.val_idx, 'r')
  targets = []
  for line in f_val:
    line_list = [item.strip() for item in line.strip('\r\n').split(',')]
    targets.append(int(line_list[2]))
  f_val.close()
 
  targets = torch.LongTensor(targets)
  targets_onehot = Variable(convert_to_one_hot(targets, 3))
  
  scores = Variable(torch.FloatTensor(scores))
  
  log_loss = KaggleLogLoss()
  log_loss_res = log_loss(scores, targets_onehot)
  print('{}, log loss: {}'.format(args.score_file, log_loss_res.data[0]))

