'''
resize the images to (256,256), the new images should be stored at
data/resized_data/
the train, additional, test folder structures should not be broken

author: Shi Yan, Gu Wang
'''
from __future__ import division, absolute_import, print_function
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from multiprocessing import Pool, cpu_count

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
  '''if filename has any of the IMG_EXTENSIONS, return True'''
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

dest_type='jpg'
img_size=(256,256)

def resize_convert(path_dir):
  path, dest_subdir = path_dir
  if is_image_file(path):
    try:
      img = Image.open(path)
      img = img.resize(img_size, Image.ANTIALIAS)  # time-consuming
      img_name = os.path.basename(path)[:-3] + dest_type
      path = os.path.join(dest_subdir, img_name)
      img.save(path)
    except:
      print('bad image: %s' % path_dir[0])
  else:
    print('{}: not image file.'.format(path_dir[0]))

def resize_convert_all(database_dir, dest_dir=None):
  """
  Resize all imgs to img_size, write into new folder, improve speed of loading data.
  Convert different types of img to dest type: jpg, png, etc.
  """
  dest_dir = dest_dir or os.path.join(os.path.dirname(database_dir),
                                      os.path.basename(database_dir) + '_' + str(img_size[0]))
  if not os.path.exists(dest_dir):
    print('making new dir for converted imgs: {}\nConverting... please wait'.format(dest_dir))
    os.makedirs(dest_dir)
  else:
    print('dest_type files already exist. Overriding!')
    # return None, dest_dir

  directories = []
  for filename in os.listdir(database_dir):
    path = os.path.join(database_dir, filename)
    if os.path.isdir(path):
      directories.append(path)
      try:
        os.makedirs(os.path.join(dest_dir, filename))
      except OSError:
        pass

  for directory in directories:
    dest_subdir = os.path.join(dest_dir, os.path.basename(directory))
    file_list = []
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      if is_image_file(path):
        file_list.append((path, dest_subdir))

    p = Pool(cpu_count())
    p.map(resize_convert, file_list)
    print('subdir done: {}'.format(dest_subdir))

  print('all imgs written to dir: {}'.format(dest_dir))
  return dest_dir

if __name__ == '__main__':
  #train_database_dir = '../data/IntelMobileODT_Cervical_Cancer_Screening/train/'
  #train_dest_dir = '../data/resized_data/train/'
  #resize_convert_all(train_database_dir, train_dest_dir)

  #additional_database_dir = '../data/IntelMobileODT_Cervical_Cancer_Screening/additional/'
  #additional_dest_dir = '../data/resized_data/additional/'
  #resize_convert_all(additional_database_dir, additional_dest_dir)

  #test_database_dir = '../data/IntelMobileODT_Cervical_Cancer_Screening/test/'
  #test_dest_dir = '../data/resized_data/test/'
  #resize_convert_all(test_database_dir, test_dest_dir)

  test_stg2_srcdir = '/data/wanggu/datasets/IntelMobileODT_Cervical_Cancer_Screening/test_stg2/'
  test_stg2_dstdir = '../data/resized_data/test_stg2/'
  resize_convert_all(test_stg2_srcdir, test_stg2_dstdir)

