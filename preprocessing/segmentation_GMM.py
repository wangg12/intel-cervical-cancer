'''
This code is refered to https://www.kaggle.com/chattob/intel-mobileodt-cervical-cancer-screening/cervix-segmentation-gmm/notebook
'''
import matplotlib.pyplot as plt
# matplotlib inline
import numpy as np
import pandas as pd
import cv2
import math

import argparse

from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from subprocess import check_output
from tqdm import tqdm


def get_image_data(image_path):
    """
    Method to get image data as np.array specifying image id and type
    """
    img = cv2.imread(image_path)
    assert img is not None, "Failed to read image : %s" % (image_path)

    return img


def maxHist(hist):
    maxArea = (0, 0, 0)
    height = []
    position = []
    for i in range(len(hist)):
        if (len(height) == 0):
            if (hist[i] > 0):
                height.append(hist[i])
                position.append(i)
        else:
            if (hist[i] > height[-1]):
                height.append(hist[i])
                position.append(i)
            elif (hist[i] < height[-1]):
                while (height[-1] > hist[i]):
                    maxHeight = height.pop()
                    area = maxHeight * (i-position[-1])
                    if (area > maxArea[0]):
                        maxArea = (area, position[-1], i)
                    last_position = position.pop()
                    if (len(height) == 0):
                        break
                position.append(last_position)
                if (len(height) == 0):
                    height.append(hist[i])
                elif(height[-1] < hist[i]):
                    height.append(hist[i])
                else:
                    position.pop()
    while (len(height) > 0):
        maxHeight = height.pop()
        last_position = position.pop()
        area =  maxHeight * (len(hist) - last_position)
        if (area > maxArea[0]):
            maxArea = (area, len(hist), last_position)
    return maxArea


def maxRect(img):
    maxArea = (0, 0, 0)
    addMat = np.zeros(img.shape)
    for r in range(img.shape[0]):
        if r == 0:
            addMat[r] = img[r]
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
        else:
            addMat[r] = img[r] + addMat[r-1]
            addMat[r][img[r] == 0] *= 0
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
    return (int(maxArea[3]+1-maxArea[0]/abs(maxArea[1]-maxArea[2])), maxArea[2], maxArea[3], maxArea[1], maxArea[0])


def cropCircle(img):
    '''
    there many imaged taken thresholded, which means many images is
    present as a circle with black surrounded. This function is to
    find the largest inscribed rectangle to the thresholed image and
    then crop the image to the rectangle.

    input: img - the cv2 module

    return: img_crop, rectangle, tile_size
    '''
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    img = cv2.resize(img, dsize=tile_size)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]

    ff = np.zeros((gray.shape[0],gray.shape[1]), 'uint8')
    cv2.drawContours(ff, main_contour, -1, 1, 15)
    ff_mask = np.zeros((gray.shape[0]+2,gray.shape[1]+2), 'uint8')
    cv2.floodFill(ff, ff_mask, (int(gray.shape[1]/2), int(gray.shape[0]/2)), 1)

    rect = maxRect(ff)
    rectangle = [min(rect[0],rect[2]), max(rect[0],rect[2]), min(rect[1],rect[3]), max(rect[1],rect[3])]
    img_crop = img[rectangle[0]:rectangle[1], rectangle[2]:rectangle[3]]
    cv2.rectangle(ff,(min(rect[1],rect[3]),min(rect[0],rect[2])),(max(rect[1],rect[3]),max(rect[0],rect[2])),3,2)

    return [img_crop, rectangle, tile_size]


def Ra_space(img, Ra_ratio, a_threshold):
    '''
    Extract the Ra features by converting RGB to LAB space.
    The higher is a value, the "redder" is the pixel.
    '''
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w*h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w/2-i)*(w/2-i) + (h/2-j)*(h/2-j))
            Ra[i*h+j, 0] = R
            Ra[i*h+j, 1] = min(imgLab[i][j][1], a_threshold)

    Ra[:,0] /= max(Ra[:,0])
    Ra[:,0] *= Ra_ratio
    Ra[:,1] /= max(Ra[:,1])

    return Ra


def get_and_crop_image(image_path):
    '''
    Input: image_path: the absolute file path of the input image

    Return: the rectangle

    TODO: add more comments and rename the variable for the code
    is hard to read
    '''

    # get image
    img = get_image_data(image_path)

    initial_shape = img.shape

    # TODO: review cropCircle
    [img, rectangle_cropCircle, tile_size] = cropCircle(img)

    # convert RGB to LAB and get the Ra space value
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = Ra_space(imgLab, 1.0, 150)
    a_channel = np.reshape(Ra[:,1], (w,h))

    # cluster with Gaussian Mixture model
    g = mixture.GaussianMixture(n_components = 2, covariance_type = 'diag', random_state = 0, init_params = 'kmeans')
    image_array_sample = shuffle(Ra, random_state=0)[:1000] # why only use 1000 point?
    g.fit(image_array_sample)
    labels = g.predict(Ra)
    labels += 1 # Add 1 to avoid labeling as 0 since regionprops ignores the 0-label.

    # The cluster that has the highest a-mean is selected.
    labels_2D = np.reshape(labels, (w,h))
    gg_labels_regions = measure.regionprops(labels_2D, intensity_image = a_channel) # what is use of this function ?
    gg_intensity = [prop.mean_intensity for prop in gg_labels_regions]
    cervix_cluster = gg_intensity.index(max(gg_intensity)) + 1

    # form a mask for cervix(255 for foreground, and 0 for background)
    mask = np.zeros((w * h,1),'uint8')
    mask[labels==cervix_cluster] = 255
    mask_2D = np.reshape(mask, (w,h))

    #
    cc_labels = measure.label(mask_2D, background=0)
    regions = measure.regionprops(cc_labels)
    areas = [prop.area for prop in regions]

    regions_label = [prop.label for prop in regions]
    largestCC_label = regions_label[areas.index(max(areas))]
    mask_largestCC = np.zeros((w,h),'uint8')
    mask_largestCC[cc_labels==largestCC_label] = 255

    img_masked = img.copy()
    img_masked[mask_largestCC==0] = (0,0,0)
    img_masked_gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY);

    _,thresh_mask = cv2.threshold(img_masked_gray,0,255,0)

    kernel = np.ones((11,11), np.uint8)
    thresh_mask = cv2.dilate(thresh_mask, kernel, iterations = 1)
    thresh_mask = cv2.erode(thresh_mask, kernel, iterations = 2)
    _, contours_mask, _ = cv2.findContours(thresh_mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours_mask, key = cv2.contourArea, reverse = True)[0]
    cv2.drawContours(img, main_contour, -1, 255, 3)

    x,y,w,h = cv2.boundingRect(main_contour)

    rectangle = [x+rectangle_cropCircle[2],
                 y+rectangle_cropCircle[0],
                 w,
                 h,
                 initial_shape[0],
                 initial_shape[1],
                 tile_size[0],
                 tile_size[1]]

    return rectangle

def parallelize_image_cropping(root, save_dir, relative_path_list):
    # TODO: implete the parallelized version

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for path in tqdm(relative_path_list):
        image_path = os.path.join(root, path)
        save_path = os.path.join(save_dir, path)

        rectangle = get_and_crop_image(image_path)

        img = get_image_data(image_path)
        if(img.shape[0] > img.shape[1]):
            tile_size = (int(img.shape[1]*256/img.shape[0]), 256)
        else:
            tile_size = (256, int(img.shape[0]*256/img.shape[1]))
        img = cv2.resize(img, dsize=tile_size)

        img = img[rectangle[1]:rectangle[1]+rectangle[3], rectangle[0]:rectangle[0]+rectangle[2]]
        # save the image
        save_dir_i = os.path.dirname(save_path)
        if not os.path.exists(save_dir_i):
            os.makedirs(save_dir_i)
        cv2.imwrite(save_path, img)
        # TODO: write the rectangle with the file name into a csv file

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', default='/data/wanggu/datasets/IntelMobileODT_Cervical_Cancer_Screening/', help='path to dataset root folder')
    parser.add_argument('--test_idx',  default='../data_utils/test.csv', help='path to test idx file, does not contain label')
    parser.add_argument('--train_idx', default='../data_utils/train_idx_0505.csv',
                    help='path to train idx file, set to ./data_utils/val_idx.csv to do quick debug the train process')
    parser.add_argument('--val_idx',   default='../data_utils/val_idx_0505.csv', help='path to val idx file')

    parser.add_argument('--workers',    type=int, default=4,   help='number of data loading workers')

    parser.add_argument('--save_dir', default='../data/segmented' )

    args = parser.parse_args()

    # get all realative file path of image data from the idx file
    relative_path_list = []
    complete_path_list = []
    with open(args.train_idx, 'r') as csvfile:
        for line in csvfile:
            relative_path = line.split(',')[0]
            relative_path_list.append(relative_path)
            complete_path = os.path.join(args.data_root, relative_path)
            complete_path_list.append(complete_path)
    with open(args.val_idx, 'r') as csvfile:
        for line in csvfile:
            relative_path = line.split(',')[0]
            relative_path_list.append(relative_path)
            complete_path = os.path.join(args.data_root, relative_path)
            complete_path_list.append(complete_path)
    with open(args.test_idx, 'r') as csvfile:
        for line in csvfile:
            relative_path = line.split(',')[0]
            relative_path_list.append(relative_path)
            complete_path = os.path.join(args.data_root, relative_path)
            complete_path_list.append(complete_path)

    # crop all the images in relative_path_list
    parallelize_image_cropping(args.data_root, args.save_dir, relative_path_list)


