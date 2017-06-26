'''
generate txt file contains images and labels
'''
import os
import os.path
import sys
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

data_root = '../data/IntelMobileODT_Cervical_Cancer_Screening/'


train_file = 'train_total.csv'
additional_file = 'additional.csv'
train_additional_file = 'train_additional.csv'
test_file = 'test.csv'
# path_to_image_from_data_root,label,int_label
# e.g.: train/Type_1/0.jpg,Type_1,0


def is_image_file(filename):
  '''if filename has any of the IMG_EXTENSIONS, return True'''
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


f_train = open(train_file, 'w')
f_additional = open(additional_file, 'w')
f_tr_add = open(train_additional_file, 'w')
f_test = open(test_file, 'w')

type_int = {'Type_1':0, 'Type_2':1, 'Type_3':2}

for f in glob.iglob(data_root+'**', recursive=True):
  if is_image_file(f):
    f = f.replace(data_root, '')
    if 'train' in f or 'additional' in f:
      type_str = f.split('/')[-2]
      f_tr_add.write('{0},{1},{2}\n'.format(f, type_str, type_int[type_str]))
      if 'train' in f:
        f_train.write('{0},{1},{2}\n'.format(f, type_str, type_int[type_str]))
      if 'additional' in f:
        f_additional.write('{0},{1},{2}\n'.format(f, type_str, type_int[type_str]))
    if 'test' in f:
      type_str=''
      int_label = ''
      f_test.write('{0},{1},{2}\n'.format(f, type_str, int_label))

f_train.close()
f_additional.close()
f_tr_add.close()
f_test.close()

