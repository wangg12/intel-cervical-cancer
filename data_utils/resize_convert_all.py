import os
from PIL import Image

def resize_convert_all(database_dir, dest_dir=None, img_size=(256,256), dest_type='jpg'):
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
    print('dest_type files already exist. Exiting without re-creating them.')
    return None, dest_dir

  directories = []
  for filename in os.listdir(database_dir):
    path = os.path.join(database_dir, filename)
    if os.path.isdir(path):
      directories.append(path)
      os.makedirs(os.path.join(dest_dir, filename))

  raw_img_types = set()
  num_total = 0
  for directory in directories:
    num_total_in_class = 0
    dest_subdir = os.path.join(dest_dir, os.path.basename(directory))
    for filename in os.listdir(directory):
      suffix = filename.split('.')[-1]
      raw_img_types.add(suffix)
      path = os.path.join(directory, filename)
      try:
        img = Image.open(path)
        img = img.resize(img_size, Image.ANTIALIAS) # time-consuming
        num_total_in_class += 1
        img_name = filename[:-len(suffix)] + dest_type
        path = os.path.join(dest_subdir, img_name)
        img.save(path)
      except:
        print('bad image: %s' % path)
    print('{} imgs written to subdir: {}'.format(num_total_in_class, dest_subdir))
    num_total += num_total_in_class

  print('{} imgs of type {} written to dir: {}'.format(num_total, dest_type, dest_dir))
  return num_total, dest_dir

if __name__ == '__main__':
  database_dir = '/data/sml/intel-cervical-cancer/data/IntelMobileODT_Cervical_Cancer_Screening/train'
  dest_dir = '/data/sml/intel-cervical-cancer/data/train_256'
  resize_convert_all(database_dir, dest_dir)
