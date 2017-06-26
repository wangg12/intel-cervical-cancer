'''
Adapt from torchvision's image folder,
use image list file to assign train/val/test split.
Here the test image list file has no label
author: Gu Wang
'''

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
  classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
  classes.sort()
  class_to_idx = {classes[i]: i for i in range(len(classes))}
  return classes, class_to_idx


def make_dataset(dir, class_to_idx):
  '''
  returns list of (image, label)
  '''
  images = []
  for target in os.listdir(dir):
    d = os.path.join(dir, target)
    if not os.path.isdir(d):
      continue

    for root, _, fnames in sorted(os.walk(d)):
      for fname in fnames:
        if is_image_file(fname):
          path = os.path.join(root, fname)
          item = (path, class_to_idx[target])
          images.append(item)

  return images


def make_dataset_from_idx_file(idx_file):
  '''
  Returns:
    images: list of (image, label),
            if no label is available (e.g. the test index file),
            just return list of (image, None)
    class_to_idx: dict of {class_str: int_label}
  '''
  class_to_idx = {}
  class_count = {} # counter of different types/classes
  f = open(idx_file, 'r')
  images = []
  for line in f:
    line_list = [item.strip() for item in line.strip('\r\n').split(',')]
    relative_path = line_list[0]
    if line_list[2] == '':
      target = None
    else:
      target = int(line_list[2])
      class_str = line_list[1]
      if not class_str in class_to_idx.keys():
        class_to_idx[class_str] = target
        class_count[target] = 0
      class_count[target] += 1

    images.append((relative_path, target))
  f.close()
  return (images, class_to_idx, class_count)



def default_loader(path):
  return Image.open(path).convert('RGB')

class MyImageFolder(data.Dataset):
  def __init__(self, root, idx_file, transform=None, target_transform=None,
               seg_transform=None, seg_root=None, loader=default_loader):
    '''
    root: dataset root dir
    idx_file: path to image label list file.
                      stored: image_path_from_data_root, str_label, int_label
    loader: method to load image
    '''
    imgs, class_to_idx, class_count = make_dataset_from_idx_file(idx_file) # list of (img, label)
    classes = list(class_to_idx.keys())
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in {}".format(idx_file)))

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.class_count = class_count
    self.total = len(imgs)
    self.transform = transform
    self.seg_transform = seg_transform
    self.target_transform = target_transform
    self.seg_root = seg_root
    self.loader = loader

  def __getitem__(self, index, ret_path=False):
    relative_path, target = self.imgs[index]
    img = self.loader(os.path.join(self.root, relative_path))
    if ret_path:
      img_name = relative_path
    else:
      img_name = relative_path.split('/')[-1]

    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      # attention: target may be None
      target = self.target_transform(target)

    if self.seg_root == None:
      return img, target, img_name
    else:
      img_seg = self.loader(os.path.join(self.seg_root, relative_path))
      if self.seg_transform is not None:
        img_seg = self.seg_transform(img_seg)
      return img, img_seg, target, img_name


  def __len__(self):
    return self.total

  def get_sample_weights(self):
    """ Return vec of sample_weight for each img, assign bad img weight to 0."""
    target_to_weight = {}
    for target, count in self.class_count.items():
      target_to_weight[target] = self.total / count

    sample_weights = []
    for _, target in self.imgs:
      sample_weights.append(target_to_weight[target])

    return sample_weights
