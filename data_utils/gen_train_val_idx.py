'''
Generate 'train.csv' and 'val.csv'
  according to 'train_additional.csv' and 'bad_images.csv'.
Some of images in 'bad_images.csv' should be ignored, some may be used.

Properly split the train/val set such that
  the number of samples of each class is  properly treated.

author: Gu Wang
'''
import os
import sys
import random
# import numpy as np
from pprint import pprint

random.seed(1234)


train_add_file = 'train_additional.csv'
bad_images_file = 'bad_images.csv'

train_idx_file = 'train_idx.csv'
val_idx_file = 'val_idx.csv'


bad_images = {}
# {img_path_1: '0_byte', img_path_2:'truncated'}
f_bad = open(bad_images_file, 'r')
for line in f_bad:
  line_list = [item.strip() for item in line.strip('\r\n').split(',')]
  bad_type = line_list[1]
  bad_img_path = line_list[0]
  if not bad_img_path in bad_images.keys():
    bad_images[bad_img_path] = bad_type
  else:
    print('Bad image duplicated!!!')

pprint(bad_images)
  
## build a clean train list by ignoring some specific bad images
# '0_byte', 'truncated'
ignore_types = ['0_byte', 'not_cervix', 'truncated'] # modify this line to ignore more types

clean_train_list = []
i = 0
with open(train_add_file, 'r') as f_all:
  for line in f_all:
    i += 1
    line_list = [item.strip() for item in line.strip('\r\n').split(',')]
    img_path = line_list[0]
    # if the img_path in any of the ignore_types
    if img_path in bad_images.keys():
      if bad_images[img_path] in ignore_types:
        print('ignore: {0}, bad type: {1}'.format(img_path, bad_images[img_path]))
        continue
    
    clean_train_list.append(line.strip('\r\n'))

print('Total number of train additional images: {}'.format(i))

## split train/val list, write to file
N_all = len(clean_train_list)
print('Number of clean train images: {}'.format(N_all))

index = [i for i in range(N_all)]
random.shuffle(index)

n_val = N_all//10
n_train = N_all - n_val

train_inds = index[0:n_train]
val_inds = index[n_train:]
print('Number of split train images: {}'.format(n_train))
print('Number of split val images: {}'.format(n_val))

train_images = [clean_train_list[idx] for idx in train_inds]
val_images = [clean_train_list[idx] for idx in val_inds]

## write to file


with open(train_idx_file, 'w') as f_train:
  for item in train_images:
    f_train.write(item+'\n')

with open(val_idx_file, 'w') as f_val:
  for item in val_images:
    f_val.write(item+'\n')

print('done.')




