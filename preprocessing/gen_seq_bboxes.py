# Use dynamic programming to connect candidate bounding boxes into a smooth box sequence in video
import cv2
from numpy.core.numeric import NaN
import numpy as np
from numpy.lib.type_check import nan_to_num
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
import os
import json
import time
import math
def N_dis(bbox1, bbox2):

    bbox1 = np.array(list(bbox1))
    bbox2 = np.array(list(bbox2))

    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]

    delta = []
    delta.append( (bbox2[0] - bbox1[0]) / w1 )
    delta.append( (bbox2[1] - bbox1[1]) / h1 )
    delta.append( math.log(w2 / w1) )
    delta.append( math.log(h2 / h1) )
    
    N_dis = sum([delta[i]**2 for i in range(len(delta))])


    return N_dis


def gen_seq_bboxes(bboxes, length):
    bbox_feedback = []

    bbox_index = 0
    bbox_found_num = 0
    bbox_not_random = []
    for frame_index in range(length): ## Count which frames have valid frames

        bboxs = bboxes[bbox_index]
        # Do statistics for the number of frames having valid candidate boxes
        if not np.isnan(bboxs).any():
            bbox_found_num += 1
            # Cache all candidate boxes and their frame index
            bbox_not_random.append((bboxs, frame_index))
        bbox_index += 1
    if len(bbox_not_random) == 0:
        return None, None, None, None, None
    # Now begin to use dynamic programming to find the optimal path for bbox sequences

    # Reward for appending new candidate bbox
    bbox_reward = -0.5
    # Minimum distance from virtual node -1 to node i
    min_distance_dp = [[bbox_reward] * len(bbox_not_random[0][0])]
    # Last bbox index to achieve minimum path distance (namely maximum reward)
    last_bbox_cut = [[(-1, -1)] * len(bbox_not_random[0][0])]
    # Maximum gap for frame connection
    max_dp_gap = 100

    # The main DP loop
    for nr_index in range(1, len(bbox_not_random)):
        bboxs, frame_index = bbox_not_random[nr_index]
        min_distance_dp_this = []
        last_bbox_cut_this = []
        for bbox in bboxs:
            # If directly connect from virtual node -1
            min_distance = bbox_reward
            min_distance_index = (-1, -1)
            for dp_index in range(max(0, nr_index-max_dp_gap), nr_index):  # find from the past 100 frames 
                last_bboxs, last_frame_index = bbox_not_random[dp_index]
                for sub_index in range(len(last_bboxs)):
                    last_bbox = last_bboxs[sub_index]
                    iou_cle = N_dis(bbox, last_bbox)
                    iou_reward = iou_cle
                    distance = min_distance_dp[dp_index][sub_index] + iou_reward + bbox_reward
                    # Record the selected path
                    if distance <= min_distance:
                        min_distance = distance
                        min_distance_index = (dp_index, sub_index)
            # Record middle results for DP path
            min_distance_dp_this.append(min_distance)
            last_bbox_cut_this.append(min_distance_index)
        # Record middle results for DP path
        min_distance_dp.append(min_distance_dp_this)
        last_bbox_cut.append(last_bbox_cut_this)

    # Now find the last bbox in sequence (the candidate box where the maximum reward path ends)
    last_index = (len(bbox_not_random)-1, 0)
    min_distance = min_distance_dp[last_index[0]][last_index[1]]
    for nr_index in range(len(bbox_not_random) - 1, -1, -1):
        for sub_index in range(len(bbox_not_random[nr_index][0])):
            if min_distance_dp[nr_index][sub_index] <= min_distance:
                last_index = (nr_index, sub_index)
                min_distance = min_distance_dp[nr_index][sub_index]

    # Now track back the selected candidate boxes in the maximum reward path to form a box sequence
    picked_bbox = []
    while last_index[1] != -1:
        bboxs, frame_index = bbox_not_random[last_index[0]]
        picked_bbox.insert(0, (bboxs[last_index[1]], frame_index))
        last_index = last_bbox_cut[last_index[0]][last_index[1]]


    # Now begin to smooth the sequence
    last_already_generated = -1
    # The list for all DP-picked frame index
    picked_frame_index = []

    for bbox_picked_index in range(len(picked_bbox)):

        bbox, frame_index = picked_bbox[bbox_picked_index]
        picked_frame_index.append(frame_index)

        # Now begin to smooth the bbox sequence in a video
        # Case 1 : index from last_gen + 1 to frame_index - 1 (candidate boxes in these frames are not selected by DP)
        for j in range(last_already_generated + 1, frame_index):
            if bbox_picked_index == 0:
                # Starting frames before the first DP-selected candidate box
                if min(list(bbox)) < 75:
                    bbox_perturbed = bbox
                else:
                    # Add very small random perturbation (optional)
                    bbox_perturbation = np.random.uniform(-3, 3, size=4)
                    bbox_perturbed = (bbox[0] + bbox_perturbation[0],
                                      bbox[1] + bbox_perturbation[1],
                                      bbox[2] + bbox_perturbation[2],
                                      bbox[3] + bbox_perturbation[3])
                bbox_feedback.append(bbox_perturbed)
            else:
                # Linear interpolation for generating the remaining boxes
                last_bbox, _ = picked_bbox[bbox_picked_index - 1]

                ratio = (frame_index - j) / (frame_index - last_already_generated)

                current_bbox = (last_bbox[0] * ratio + bbox[0] * (1 - ratio),
                                last_bbox[1] * ratio + bbox[1] * (1 - ratio),
                                last_bbox[2] * ratio + bbox[2] * (1 - ratio),
                                last_bbox[3] * ratio + bbox[3] * (1 - ratio))
                bbox_feedback.append(current_bbox)

        # Case 2 : index equals to frame_index (the current frame has a candidate box selected by DP)
        bbox_feedback.append(bbox)
        last_already_generated = frame_index

    # Fill in the last bboxes
    pending_num = length - len(bbox_feedback)
    last_bbox = bbox_feedback[-1]
    # Ending frames after the last DP-selected candidate box
    for i in range(pending_num):
        if min(list(last_bbox)) < 50:
            bbox_perturbed = last_bbox
        else:
            # Add very small random perturbation (optional)
            bbox_perturbation = np.random.uniform(-3, 3, size=4)
            bbox_perturbed = (last_bbox[0] + bbox_perturbation[0],
                              last_bbox[1] + bbox_perturbation[1],
                              last_bbox[2] + bbox_perturbation[2],
                              last_bbox[3] + bbox_perturbation[3])
        bbox_feedback.append(bbox_perturbed)

    assert length == len(bbox_feedback)

    # Now do statistics and calculate various related metrics
    # Average box vary in box sequence (not utilized at last, deprecated)
    total_vary = 0
    for i in range(length - 1):
        current_bbox = bbox_feedback[i]
        next_bbox = bbox_feedback[i + 1]
        for j in range(len(current_bbox)):
            total_vary += abs(current_bbox[j] - next_bbox[j])

    aver_vary = total_vary / (length - 1)

    # Frequency for candidate bboxes be picked by dp
    bbox_picked_freq = len(picked_bbox) / len(bboxes)

    bbox_found_freq = bbox_found_num / len(bboxes)

    return bbox_feedback, picked_frame_index, bbox_found_freq, bbox_picked_freq, aver_vary

# Calculate frame quality score, important for filtering boxes of high quality
def calc_nearby_bbox_freq(picked_frame_index, video_length, search_range=None):

    if search_range is None or len(search_range) == 0:
        search_range = [3, 10]

    search_range = [s for s in search_range]

    # Init for doing statistics
    freq_dicts = [[0] * video_length for _ in range(len(search_range))]
    freq_collect_max = [[0] * video_length for _ in range(len(search_range))]

    # Do statistic for the number of adjacent frames (of a certain frame) potential to be selected by DP
    for r_i in range(len(search_range)):
        for v_i in range(1, video_length):
            left_index = max(0, v_i - search_range[r_i])
            right_index = min(video_length - 1, v_i + search_range[r_i])
            for sub_i in range(left_index, right_index + 1):
                # increment count
                current = freq_collect_max[r_i][sub_i]
                freq_collect_max[r_i][sub_i] = current + 1

    # Do statistic for the number of adjacent frames (of a certain frame) indeed selected by DP
    for r_i in range(len(search_range)):
        for v_i in picked_frame_index:
            left_index = max(0, v_i - search_range[r_i])
            right_index = min(video_length - 1, v_i + search_range[r_i])
            for sub_i in range(left_index, right_index + 1):
                # increment count
                current = freq_dicts[r_i][sub_i]
                freq_dicts[r_i][sub_i] = current + 1

    # Calculate the frequency of DP selection within all adjacent frames (of a certain frame)
    feedback = []
    for v_i in range(video_length):
        score = []
        for r_i in range(len(search_range)):
            if freq_collect_max[r_i][v_i] != 0:
                score.append(float(freq_dicts[r_i][v_i] / freq_collect_max[r_i][v_i]))
            else:
                score.append(float(0.0))
        feedback.append(score)
    return feedback


if __name__ == '__main__':
    path = 'coarse_boxes/'# 'coarse_boxes/'
    img_data = '/YOUR/PATH/NAT2021-train/train_clip/' # your path to original seq
    save_path = 'pseudo_anno' # generated pseudo annotations will be saved here
    coarse_bbox_files = glob(os.path.join(path, '*.npy'))
    coarse_bbox_files.sort()
    s = []
    for coarse_bbox in tqdm(coarse_bbox_files):
        v = dict()
        current_seq = coarse_bbox.split('/')[-1][:-4]
        max_bboxes = np.load(coarse_bbox, allow_pickle= True) 
        seq_bboxs, picked_frame_index, bbox_found_freq, bbox_picked_freq, aver_vary = \
                    gen_seq_bboxes(max_bboxes, length=len(max_bboxes))
        if seq_bboxs:
            freq_dict = calc_nearby_bbox_freq(picked_frame_index, video_length=len(seq_bboxs),
                                search_range=[3, 10])
            print("bbox_found_freq: {}, bbox_picked_freq: {}, vary_aver: {}, consumed time: {} seconds.".format(
                            bbox_found_freq, bbox_picked_freq, aver_vary, 0))

            v['base_path'] = current_seq
            v['frame'] = []
            v["aver_vary"] = aver_vary
            v["bbox_found_freq"] = bbox_found_freq
            v["bbox_picked_freq"] = bbox_picked_freq
            v['picked_frame_index'] = picked_frame_index
            img_index = os.listdir(os.path.join(img_data, current_seq))
            img_index.sort()
            
            img_temp = cv2.imread(os.path.join(img_data, v['base_path'], img_index[0]))
            img_size = img_temp.shape
            for i in range(len(max_bboxes)):
                f = dict()
                f['frame_sz'] = [int(img_size[1]), int(img_size[0])]
                f['img_path'] = img_index[i]
                o = dict()
                o['trackid'] = 0
                o['bbox'] = [int(seq_bboxs[i][0]*f['frame_sz'][0]/448), int(seq_bboxs[i][1]*f['frame_sz'][1]/448), \
                    int(seq_bboxs[i][2]*f['frame_sz'][0]/448), int(seq_bboxs[i][3]*f['frame_sz'][1]/448)]

                o['confidence'] = freq_dict[i]
                f['objs'] = [o]
                v['frame'].append(f)
            
            if not os.path.isdir('pseudo_anno'): os.mkdir('pseudo_anno')
            with open("{}/{}_gt.txt".format(save_path, current_seq),"w") as f:
                for i in v['frame']:
                    f.write(str(i['objs'][0]['bbox'])[1:-1] + '\n') 

            vis = False # set 'True' if you want to see visualization results
            if vis:
                i = 0
                while True:
                    if i >= len(max_bboxes):
                        i = 0
                    bbox = seq_bboxs[i]
                    image = cv2.imread(os.path.join(img_data, v['base_path'], v['frame'][i]['img_path']))
                    draw = cv2.rectangle(image, (v['frame'][i]['objs'][0]['bbox'][0], v['frame'][i]['objs'][0]['bbox'][1]), (v['frame'][i]['objs'][0]['bbox'][2], v['frame'][i]['objs'][0]['bbox'][3]),
                                            (0, 255, 0), 1)
                    i += 1
                    
                    cv2.imshow("Frame", draw)
                    time.sleep(0.02)
                    key = cv2.waitKey(1) & 0xFF
                    # If the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break