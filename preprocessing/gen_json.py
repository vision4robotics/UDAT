#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import json
from os.path import join, exists
import os
from tqdm import tqdm
# import pandas as pd

dataset_path = '/YOUR/PATH/NAT2021-train/train_clip/'
pseudo_label_path = 'pseudo_anno' # path to generated pseudo label
def parse_and_sched(dl_dir='.'):
    js = {}

    videos = os.listdir(os.path.join(dataset_path))
    for video in tqdm(videos):
        if video == 'list.txt':
            continue
        gt_path = join(pseudo_label_path, video+ '_gt.txt')
        f = open(gt_path, 'r')
        groundtruth = f.readlines()
        f.close()
        idx_woOcc = 0 
        for idx, gt_line in enumerate(groundtruth):
            if gt_line == 'NaN,NaN,NaN,NaN\n':
                continue
            gt_image = gt_line.strip().split(',')
            frame = '%06d' % (int(idx_woOcc)) # idx
            obj = '%02d' % (int(0))
            bbox = [int(float(gt_image[0])), int(float(gt_image[1])),
                    int(float(gt_image[0])) + int(float(gt_image[2])),
                    int(float(gt_image[1])) + int(float(gt_image[3]))]  # xmin,ymin,xmax,ymax

            if video not in js:
                js[video] = {}
            if obj not in js[video]:
                js[video][obj] = {}
            js[video][obj][frame] = bbox
            idx_woOcc = idx_woOcc + 1

    json.dump(js, open('/YOUR/PATH/NAT2021-train/train.json', 'w'), indent=4, sort_keys=True)

    print(': All videos downloaded' )


if __name__ == '__main__':
    parse_and_sched()