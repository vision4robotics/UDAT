from __future__ import absolute_import, division, print_function
from genericpath import isdir
import os
from numpy.core.numeric import NaN

import torch
from torch.utils import data
from torchvision.transforms import functional as TF
import numpy as np

import argparse
from tqdm import tqdm

from libs.datasets import get_transforms, get_datasets
from libs.networks import VideoModel
from libs.utils.pyt_utils import load_model
from draw_mask import union_image_mask
from skimage.measure import regionprops
from skimage import measure, morphology
import time
# os.environ['CUDA_VISIBLE_DEVICES']='1'
configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        spshot=20000,
        nclass=2,
        sshow=10,
    ),
    'stage2_cfg': dict(
        NUM_BRANCHES = 2,
        NUM_CHANNELS = [32, 64],
        NUM_BLOCKS = [4, 4],
    ),
    'stage3_cfg': dict(
        NUM_BRANCHES = 3,
        NUM_CHANNELS=[256, 512, 1024],
        NUM_BLOCKS=[4, 4, 4],
    ),
    'stage4_cfg': dict(
        NUM_BRANCHES = 4,
        NUM_BLOCKS = [4, 4, 4, 4],
        NUM_CHANNELS = [256, 512, 1024, 256],
    )
}

CFG = configurations
parser = argparse.ArgumentParser()

# Dataloading-related settings
parser.add_argument('--data', type=str, default= '/YOUR/PATH/NAT2021-train/train_clip_enhanced/',
                    help='path to datasets folder') # set the path of enhanced imgs
parser.add_argument('--dataset', default='NAT2021', type=str,
                    help='dataset name for inference')
parser.add_argument('--split', default='test', type=str, choices=['test',],
                    help='dataset split for inference')
parser.add_argument('--checkpoint', default='models/checkpoints/video_current_best_model.pth',
                    help='path to the pretrained checkpoint')
parser.add_argument('--dataset-config', default='config/datasets.yaml',
                    help='dataset config file')
parser.add_argument('--vis', action='store_true',default='',
        help='whether save predicted saliency maps and masked imgs')
parser.add_argument('--results_folder', default='data/results/',
                    help='location to save predicted saliency maps')

parser.add_argument('--boxes_results', default='coarse_boxes/',
                    help='location to save predicted boxes')           
parser.add_argument('-j', '--num_workers', default=1, type=int, metavar='N',
                    help='number of data loading workers.')

# Model settings
parser.add_argument('--size', default=448, type=int,
                    help='image size')
parser.add_argument('--os', default=16, type=int,
                    help='output stride.')
parser.add_argument("--clip_len", type=int, default=4,
                    help="the number of frames in a video clip.")

args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

if cuda:
    torch.backends.cudnn.benchmark = True
    current_device = torch.cuda.current_device()
    print("Running on", torch.cuda.get_device_name(current_device))
else:
    print("Running on CPU")

data_transforms = get_transforms(
    input_size=(args.size, args.size),
    image_mode=False
)
dataset = get_datasets(
    name_list=args.dataset,
    split_list=args.split,
    config_path=args.dataset_config,
    root=args.data,
    training=False,
    transforms=data_transforms['test'],
    read_clip=True,
    random_reverse_clip=False,
    label_interval=1,
    frame_between_label_num=0,
    clip_len=args.clip_len
)

dataloader = data.DataLoader(
    dataset=dataset,
    batch_size=1, # only support 1 video clip
    num_workers=args.num_workers,
    shuffle=False
)

model = VideoModel(output_stride=args.os, cfg=CFG)

# load pretrained models
if os.path.exists(args.checkpoint):
    print('Loading state dict from: {0}'.format(args.checkpoint))
    model = load_model(model=model, model_file=args.checkpoint, is_restore=True)
else:
    raise ValueError("Cannot find model file at {}".format(args.checkpoint))

model.to(device)

def saliency_to_bbox(pred, tiny_ratio=0.02):
    pred = morphology.binary_closing(pred)
    pred = morphology.remove_small_objects(pred, 80) # filter isolated small areas smaller than 80 pixels
    pred = morphology.remove_small_holes(pred, 80) # filter small holes

    cand = measure.label(pred) # mark connected domain
    w_c, h_c = pred.shape
    max_bboxs = []
    # top_n = 1
    for region in regionprops(cand):
        bbox = region.bbox
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < 100:
            continue
        max_bboxs.append((bbox[1], bbox[0], bbox[3], bbox[2]))

    return max_bboxs


def inference():
    model.eval()
    print("Begin inference on {} {}.".format(args.dataset, args.split))
    running_mae = 0.0
    running_smean = 0.0
    start = time.time()
    v = dict()
    s = []
    final = []
    max_bboxes = [] # store predicted boxes of a sequence
    for data in tqdm(dataloader):
        current_seq = data[0]['image_id'][0].split('/')[0]

        images = [frame['image'].to(device) for frame in data]
        labels = []
        for frame in data:
            labels.append(frame['label'].to(device))
        with torch.no_grad():
            preds = model(images)

        # generate pseudo label
        temp_box = []
        temp = 0
        for pred in preds:
            pred[pred<0.1] = 0 
            pred[pred>=0.1] = 1 
            pred = pred.squeeze(0).squeeze(0).cpu().numpy().astype(int)
            bboxs = saliency_to_bbox(pred)
            if not bboxs:
                bboxs = [(NaN, NaN, NaN, NaN)]
            temp_box.append(bboxs)
            index = int(data[temp]['image_id'][0].split('/')[1])

            max_bboxes.append(bboxs)
            if index == len(os.listdir(os.path.join(args.data, current_seq))):
                if not isdir(args.boxes_results): os.mkdir(args.boxes_results)
                np.save(os.path.join(args.boxes_results + current_seq), max_bboxes[:index]) 
                print('saved')
                max_bboxes = [] # reinitialize storage
                break
            temp += 1 
    
    
        # # save predicted saliency maps
        if not args.vis:
            continue
        for i, (label_, pred_) in enumerate(zip(labels, preds)):
            for j, (label, pred) in enumerate(zip(label_.detach().cpu(), pred_.detach().cpu())):
                dataset = data[i]['dataset'][j]
                image_id = data[i]['image_id'][j]
                result_path = os.path.join(args.results_folder, "{}/{}.png".format(dataset, image_id))
                
                pred[pred<0.1] = 0 
                pred[pred>=0.1] = 1 
                result = TF.to_pil_image(pred)
                dirname = os.path.dirname(result_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                result.save(result_path)
                # draw mask
                if not os.path.exists(dirname.replace('results', 'mask')):
                    os.makedirs(dirname.replace('results', 'mask'))
                union_image_mask(os.path.join(args.data, "{}.jpg".format(image_id)), result_path, result_path.replace('results', 'mask').replace('png','jpg'), temp_box[i])

if __name__ == "__main__":
    inference()
