# -*- coding: utf-8 -*-
# adapted from kaggle kernel
# author: Gu Wang

# This script checks the dataset's image shape, and jpeg file's error or warning.
# This script checks all dataset (stage 1, train, test, additional)
# This script is part 1 of 4, for computing in 1200 sec.

# Shape includes size (height, width) and number of color channel (RGB = 3)
# shape_1 = height, shape_2 = width, shape_3 = 3

# error   =  blank 0 byte file
# additional/Type_2/2845.jpg
# additional/Type_2/5892.jpg
# additional/Type_2/5893.jpg

# warning = Premature end of JPEG file
# train/Type_1/1339.jpg has about 55% data of the image size. This file can't be used.
# additional/Type_1/3068.jpg has about 78% data of the image size. This file can be used.
# additional/Type_2/7.jpg has about 75% data of the image size. This file maybe be used.

import os
import platform
import cv2
import numpy as np
from PIL import Image
import pandas
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def get_list_abspath_img(dataset_dir):
  '''Get the list of all jpeg files in directory'''
  list_abspath_img = []
  for str_name_file_or_dir in os.listdir(dataset_dir):
    if ('.jpg' in str_name_file_or_dir) == True:
      list_abspath_img.append(os.path.join(dataset_dir, str_name_file_or_dir))
  list_abspath_img.sort()
  return list_abspath_img


data_root = '../data/IntelMobileODT_Cervical_Cancer_Screening/'

dataset_dir_train_1 = data_root + 'train/Type_1'
dataset_dir_train_2 = data_root + 'train/Type_2'
dataset_dir_train_3 = data_root + 'train/Type_3'
dataset_dir_test    = data_root + 'test/test'
dataset_dir_add_1   = data_root + 'additional/Type_1'
dataset_dir_add_2   = data_root + 'additional/Type_2'
dataset_dir_add_3   = data_root + 'additional/Type_3'

# # For kaggle's kernels environment (docker container?)
# abspath_dataset_dir_train_1 = '/kaggle/input/train/Type_1'
# abspath_dataset_dir_train_2 = '/kaggle/input/train/Type_2'
# abspath_dataset_dir_train_3 = '/kaggle/input/train/Type_3'
# abspath_dataset_dir_test    = '/kaggle/input/test/'
# abspath_dataset_dir_add_1   = '/kaggle/input/additional/Type_1'
# abspath_dataset_dir_add_2   = '/kaggle/input/additional/Type_2'
# abspath_dataset_dir_add_3   = '/kaggle/input/additional/Type_3'

list_abspath_img_train_1 = get_list_abspath_img(dataset_dir_train_1)
list_abspath_img_train_2 = get_list_abspath_img(dataset_dir_train_2)
list_abspath_img_train_3 = get_list_abspath_img(dataset_dir_train_3)

list_abspath_img_test    = get_list_abspath_img(dataset_dir_test)

list_abspath_img_add_1   = get_list_abspath_img(dataset_dir_add_1)
list_abspath_img_add_2   = get_list_abspath_img(dataset_dir_add_2)
list_abspath_img_add_3   = get_list_abspath_img(dataset_dir_add_3)

list_abspath_img_train = list_abspath_img_train_1 + list_abspath_img_train_2 + list_abspath_img_train_3
list_abspath_img_add = list_abspath_img_add_1 + list_abspath_img_add_2 + list_abspath_img_add_3


# Header of output
pandas_header = ['abspath', 'shape_1', 'shape_2', 'shape_3', 'error', 'warning']
pandas_data   = []

# Join lists of abspath
list_abthpath = list_abspath_img_train + list_abspath_img_test + list_abspath_img_add

print('number of train: {}'.format(len(list_abspath_img_train)))
print('number of train Type_1: {}'.format(len(list_abspath_img_train_1)))
print('number of train Type_2: {}'.format(len(list_abspath_img_train_2)))
print('number of train Type_3: {}'.format(len(list_abspath_img_train_3)))
print('number of additional: {}'.format(len(list_abspath_img_add)))
print('number of additional Type_1: {}'.format(len(list_abspath_img_add_1)))
print('number of additional Type_2: {}'.format(len(list_abspath_img_add_2)))
print('number of additional Type_3: {}'.format(len(list_abspath_img_add_3)))
print('number of test: {}'.format(len(list_abspath_img_test)))

# Check all jpeg file
# for abspath_img in tqdm(list_abthpath):
def check_image(abspath_img):  
  img = cv2.imread(abspath_img)

  # 0 byte file can be imread(), but doesn't have shape attribute
  if hasattr(img, 'shape') == False:
    # Add shape info to pandas_data (header = abspath,shape_1,shape_2,shape_3,error,warning)
    print('error in {} , 0 byte file'.format(abspath_img))
    # check file is 0 byte: f = open(abspath_img); len(f.read())
    pandas_data.append([abspath_img, 'Nan', 'Nan', 'Nan', 'error', ''])
    
  else:
    # img.shape returns like (640, 480, 3) -> ['640', '480', '3']
    list_shape_str = [str(item) for item in img.shape]
    warning    = ''
    try:
      # Check Premature end of JPEG file
      # If premature jpeg file, TypeError: int() argument must be a string or a number, not 'JpegImageFile'
      cv2.cvtColor(np.array(Image.open(abspath_img), dtype=np.uint8), cv2.COLOR_RGB2BGR)
    except:
      # Maybe(!?) Premature end of JPEG file
      warning = 'warning'
      print('warning, maybe premature end of JPEG file in {}'.format(abspath_img))
    finally:
      # Add shape info to pandas_data (header = abspath,shape_1,shape_2,shape_3,error,warning)
      pandas_data.append([abspath_img, list_shape_str[0],
                          list_shape_str[1], list_shape_str[2], '', warning])

p = Pool(cpu_count())
p.map(check_image, list_abthpath)

pandas.DataFrame(pandas_data, columns = pandas_header).to_csv('check_all_dataset.csv', index=False)


