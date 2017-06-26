from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer
import os
import numpy as np
import cv2


def cut_det(data_dir, dest_dire, img_size=(256,256), dest_type='jpg'):
    if not os.path.exists(dest_dir):
        print('making new dir for converted imgs: {}'.format(dest_dir))
        os.makedirs(dest_dir)
    else:
        print('dest_type files already exist. Type 0 to abort, 1 to continue.')
        flag = input()
        if not flag:
            return None, dest_dir

    directories = []
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            try:
                os.makedirs(os.path.join(dest_dir, filename))
            except OSError, e:
                if e.errno != 17:
                    raise
                pass

    # # read from idx_file
    # with open(idx_file, 'r') as f:
    #     lines = f.readlines()
    #     img_paths = [os.path.join(data_dir, l.strip().split()[0]) for l in lines]

    model_file = 'models/B300_FG03/faster_rcnn_100000.h5'
    expm = model_file.split('/')[-1].split('.')[0]
    expm_dir = os.path.join('demo', expm)
    if not os.path.exists(expm_dir):
        os.makedirs(expm_dir)

    detector = FasterRCNN()
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval() # set model in evaluation mode, has effect on Dropout and Batchnorm. Use train() to set train mode.
    print('load model successfully!\nConverting... please wait')
    t = Timer()

    raw_img_types = set()
    num_total = 0
    for directory in directories:
        num_total_in_class = 0
        dest_subdir = os.path.join(dest_dir, os.path.basename(directory))
        for filename in os.listdir(directory):
            t.tic()
            suffix = filename.split('.')[-1]
            raw_img_types.add(suffix)
            path = os.path.join(directory, filename)
            try:
                img = cv2.imread(path)
                img_name = filename[:-len(suffix)] + dest_type
                dets, scores, classes = detector.detect(img, 0.7)
                if len(dets) == 0:
                    img_roi = img
                else:
                    x1, y1, x2, y2 = dets[0].astype(int)  # choose the highest score
                    img_roi = img[y1:y2, x1:x2]

                if img_size:
                    img_roi = cv2.resize(img_roi, img_size)
                path = os.path.join(dest_subdir, img_name)
                cv2.imwrite(path, img_roi)
                num_total_in_class += 1

                # # save all dets with different names
                # for i, det in enumerate(dets):
                #     det = tuple(int(x) for x in det)
                #     x1, y1, x2, y2 = det
                #     img_roi = img[y1:y2, x1:x2]
                #     if img_size:
                #         img = cv2.resize(img, img_size)
                #     if i == 0:
                #         img_name = filename[:-len(suffix)] + dest_type
                #         path = os.path.join(dest_subdir, img_name)
                #         cv2.imwrite(path, img_roi)
                #     else:
                #         img_name = filename[:-(len(suffix)+1)] + '_' + str(i) + '.' + dest_type
                #         path = os.path.join(dest_subdir, img_name)
                #         cv2.imwrite(path, img_roi)

            except:
                print('bad image: %s' % path)
            t.toc(average=True)
        print('{} imgs written to subdir: {}'.format(num_total_in_class, dest_subdir))
        num_total += num_total_in_class

    print('{} imgs of type {} written to dir: {}'.format(num_total, dest_type, dest_dir))
    print('average {}s per img'.format(t.toc(average=True)))
    return num_total, dest_dir


if __name__ == '__main__':
    data_dir = 'data/cervix/test'
    # data_dir = 'data/cervix/train'
    # data_dir = 'data/cervix/additional'
    dest_dir = 'data/cervix_roi/test'
    cut_det(data_dir, dest_dir, img_size=None)
