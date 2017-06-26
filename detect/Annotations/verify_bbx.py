import os
import cv2
import numpy as np

def plot_bbx(img_path, bbxs):
  img = cv2.imread(img_path)
  for bbx in bbxs:
    # bbx = tuple(int(x) for x in bbx)
    topleft = (bbx[0], bbx[1])
    bottomright = (bbx[0] + bbx[2], bbx[1] + bbx[3])
    cv2.rectangle(img, topleft, bottomright, (255, 205, 51), 4)
  # for display
  #img = cv2.resize(img, (256,256), interpolation = cv2.INTER_CUBIC)
  cv2.imwrite('out.jpg', img)
  # cv2.imshow('img_bbx', img)
  # cv2.waitKey(0)

if __name__ == '__main__':
  img_root = '../'

  # line = 'train/Type_1/928.jpg 2 326 1145 972 1063 1698 785 480 1658'
  line = 'train/Type_1/928.jpg 1 326 1145 972 1063'
  line_split = line.strip().split()
  img_path = os.path.join(img_root, line_split[0])
  bbx_list = [int(val) for val in line_split[2:]]
  assert len(bbx_list) % 4 == 0, 'len(bbx_list) = {}, not divisible by 4'.format(len(bbx_list))
  bbxs = zip(*[iter(bbx_list)]*4)

  #img_path = 'Type_1/0.jpg'
  #bbxs = [[882, 961, 1042, 1106]]
  print img_path, bbxs
  plot_bbx(img_path, bbxs)
