from __future__ import print_function, division
import argparse
import os
import sys
import numpy as np
from datetime import datetime

def load_scores(score_path):
  f = open(score_path, 'r')
  i = 0
  scores = []
  image_names = []
  for line in f:
    if i == 0:
      i += 1
      continue
    line_list = line.strip('\r\n').split(',')
    img_name, score_1, score_2, score_3 = line_list
    scores.append([float(score_1), float(score_2), float(score_3)])
    image_names.append(img_name)
    i += 1
  f.close()
  return np.array(scores).astype(np.float32), image_names


if __name__ == '__main__':
  # input an ensemble file list, output a single ensembled score
  # for example: ls submission/inception_v3_model_epoch_15_fold_* >ensemble_list.txt
  if len(sys.argv) < 2:
    print('usage: python xxx.py /path/to/ensemble_list.txt')
    exit()
  ensemble_file = sys.argv[1] # './submission/ensemble_list.txt'
  f = open(ensemble_file, 'r')
  ensemble_list = [line.strip('\r\n') for line in f.readlines()]
  
  file_score_dict = {}
  for line in ensemble_list:
    if not line in file_score_dict.keys():
      scores, image_names =  load_scores(line)
      file_score_dict[line] = scores 
  # ensemble, currently average, TODO: weighted or learnable
  final_scores = sum(file_score_dict.values())/float(len(file_score_dict.keys()))
  
  filename = 'ensemble_submission_{}_{}.csv'.format(ensemble_file.split('.')[0],
                  datetime.now().strftime('_%Y_%m_%d_%H_%M_%S'))
  header_ = ','.join(['image_name', 'Type_1', 'Type_2', 'Type_3'])
  with open(os.path.join('./submission', filename), 'w') as f_csv:
    f_csv.write(header_+'\n')
    for i in range(len(image_names)):
      f_csv.write('{0},{1:.8f},{2:.8f},{3:.8f}\n'.format(image_names[i],
                final_scores[i,0], final_scores[i,1], final_scores[i,2]))
