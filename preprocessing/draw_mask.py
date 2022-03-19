import cv2
import os
import numpy as np
from glob import glob
from genericpath import isdir
from numpy.core.numeric import NaN
from tqdm.contrib import tzip
def union_image_mask(image_path, mask_path, save_name, bboxes, color = (160, 32, 240)):
    image = cv2.imread(image_path)
    mask_2d = cv2.imread(mask_path,0)
    chang = image.shape[1]
    kuan = image.shape[0]
    
    mask_2d = cv2.resize(mask_2d,(chang, kuan))

    coef = 255 if np.max(image)<3 else 1
    image = (image * coef).astype(np.float32)
    contours, _ = cv2.findContours(mask_2d, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(image.shape)
    # cv2.drawContours(image, contours, -1, color, 40)
    zeros = np.zeros((image.shape), dtype=np.uint8)
    mask = cv2.fillPoly(zeros, contours, color)
    image = 0.3*mask +image
    # for box in bboxes:
    #     if not NaN in box:
    #         cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
    save_path = save_name
    save_path = save_path.replace(save_name.split('/')[-1], '/')
    if not isdir(save_path):
        os.mkdir(save_path)
    cv2.imwrite(save_name,image)

if __name__ == '__main__':
    ori_path = '/path/to/original/img'
    mask_path = '/path/to/binary/mask'
    images = glob(os.path.join(ori_path, '*.jpg'))
    images.sort()
    mask = glob(os.path.join(mask_path, '*.png'))
    mask.sort()
    for image_name, mask_name in tzip(images, mask):
        union_image_mask(image_name, mask_name, mask_name.replace('results', 'mask'))
