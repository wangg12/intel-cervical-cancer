import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer
import os


def test(visualize=False):
    import os
    im_file = 'data/cervix/train/Type_2/1381.jpg'
    im_name = im_file.split('/')[-1]
    image = cv2.imread(im_file)

    # model_file = 'models/VGGnet_fast_rcnn_iter_70000.h5'
    model_file = 'models/saved_model3/faster_rcnn_100000.h5'
    expm = model_file.split('/')[-1].split('.')[0]
    expm_dir = os.path.join('demo', expm)
    if not os.path.exists(expm_dir):
        os.makedirs(expm_dir)

    detector = FasterRCNN()
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval() # set model in evaluation mode, has effect on Dropout and Batchnorm. Use train() to set train mode.
    print('load model successfully!')

    # network.save_net(r'/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5', detector)
    # print('save model succ')

    t = Timer()
    t.tic()
    # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
    dets, scores, classes = detector.detect(image, 0.7)
    runtime = t.toc()
    print('total spend: {}s'.format(runtime))

    im2show = np.copy(image)
    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 4)
        cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    cv2.imwrite(os.path.join('demo', expm, im_name), im2show)

    if visualize:
        im2show = cv2.resize(im2show, None, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('demo', im2show)
        cv2.waitKey(0)


if __name__ == '__main__':
    test()
