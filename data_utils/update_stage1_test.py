'''
generate txt file contains images and labels
'''
from __future__ import division, absolute_import, print_function

import os
import os.path
import sys
import glob
import csv

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

data_root = '../data/resized_data/test/test/'

test_file = 'test_stg1.csv'
groundtruth_file = 'solution_stg1_release.csv'

def is_image_file(filename):
  '''if filename has any of the IMG_EXTENSIONS, return True'''
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

f_test = open(test_file, 'w')

type_int = {'Type_1':0, 'Type_2':1, 'Type_3':2}
int_str = ['Type_1', 'Type_2', 'Type_3']

# import the ground truth of test image
gt = {}
with open(groundtruth_file, 'r') as gt_file:
  reader = csv.reader(gt_file, delimiter=',', quotechar='\n')
  i = 0
  for row in reader:
    i = i+1
    if i==1:
      continue
    # print(row)
    img_name = row[0]
    label = row[1:].index('1')
    gt[img_name] = label

# print(gt)

for f in glob.glob(data_root+'*.*'):
  if is_image_file(f):
    f = f.replace(data_root, 'test/test/')
    print(f)
    if 'test' in f:
      img_name = os.path.basename(f)
      int_label = gt[img_name]
      type_str= int_str[int_label]
      f_test.write('{0},{1},{2}\n'.format(f, type_str, int_label))

f_test.close()

