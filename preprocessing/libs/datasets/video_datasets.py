from __future__ import print_function

import os
import random
import yaml
import numpy as np
from glob import glob
from PIL import Image

import torch
from torch.utils import data


class SaliencyDataset(data.Dataset):
    """Saliency Dataset Base Class
        Parameters
        ----------
            name: string
                dataset name
            image_ext: string
                image filename extension e.g. ".jpg", ".png"
            label_ext: string
                label filename extension e.g. ".jpg", ".png"
            image_dir: string
                directory name of images e.g. "JPEGImages"
            label_dir: string
                directory name of labels e.g. "Annotations"
            root: string
                path of dataset root
            split: string
                the split set name of dataset e.g. "train", "val", "test"
            training: bool
                it is True when training/validation and False when inference
            transform : callable, optional
                A function/transform that takes in
                a sample and returns a transformed version.
        Attributes
        ----------
            flies: 1d list
                get a frame information by files[frame_index]
    """

    def __init__(self, name, image_ext, label_ext, image_dir,
                 label_dir, root, split, training, transforms):

        self.name = name
        self.root = os.path.join(root, name)
        if self.name == 'NAT2021':
            self.root = os.path.join(root)
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.image_dir = os.path.join(self.root) 
        self.label_dir = os.path.join(self.root) 

        self.split = split
        # not labels in inference mode
        self.training = training
        self.transforms = transforms
        self.files = []

    def _get_frame(self, frame_info, i, j, h, w, flip_index, flag=False):
        image_path = frame_info['image_path']
        image = Image.open(image_path).convert('RGB')  # RGB format
        image_size = image.size[:2]
        item = {'dataset': self.name,
                'image_id': frame_info['image_id'],
                'height': image_size[0],
                'width': image_size[1]}

        if 'label_path' in frame_info:
            label = np.array(Image.open(frame_info['label_path']).convert('L')).astype(np.float32)
            # if label.max() > 1:
            #    label = (label >= 128).astype(np.uint8) # convert 255 to 1
            # else:
            #    label = (label >= 0.5).astype(np.uint8)
            if label.max() > 1:
                label = label / 255.0
            label = Image.fromarray(label)
        else:
            label = None

        sample = {'image': image, 'label': label}
        sample, i, j, h, w, flip_index = self.transforms(sample, i=i, j=j, h=h, w=w, flip_index=flip_index, flag=flag)
        item['image'] = sample['image']
        if label is not None:
            item['label'] = sample['label']
        return item, i, j, h, w, flip_index

    def _set_files(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.files)


class ImageDataset(SaliencyDataset):
    """Image Saliency Dataset base class
        Each index can get a image
    """

    def __init__(self, split_dir, **kwargs):
        super(ImageDataset, self).__init__(**kwargs)
        self.split_dir = split_dir
        self._set_files()

    def _set_files(self):
        # txt_fname = os.path.join(self.root, self.split_dir, self.split + "_id.txt")
        # with open(txt_fname, 'r') as f:
        #     images_id = f.read().split()
        txt_fname = os.path.join(self.root, self.split)
        images_id = os.listdir(txt_fname)
        for image_id in images_id:
            image_id = image_id[:-4]
            image_path = os.path.join(self.image_dir, image_id + self.image_ext)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            frame_info = {'image_id': image_id,
                          'image_path': image_path}
            if self.training:
                label_path = os.path.join(self.label_dir, image_id + self.label_ext)
                if not os.path.isfile(label_path):
                    raise FileNotFoundError(label_path)
                frame_info['label_path'] = label_path
            self.files.append(frame_info)

    def __getitem__(self, index):
        frame_info = self.files[index]
        return self._get_frame(frame_info)


class VideoDataset(SaliencyDataset):
    """Video Saliency Dataset base class
        Parameters
        ----------
            video_split : dict
                A dict containing the train/val/test split of dataset.
                Each value of it is a list of video name
            default_label_interval: int
                The annotations interval of this dataset which means every "default_label_interval" will be a label
                e.g. the "default_label_interval" of "VOS" dataset is 15
            label_interval : int
                Resample labels based on default_label_interval which
                means every "default_label_interval*label_interval" will be a label
            frame_between_label_num: int
                The number of frames without label between two frames with label.
    """

    def __init__(self, video_split, default_label_interval, label_interval, frame_between_label_num, **kwargs):
        super(VideoDataset, self).__init__(**kwargs)

        self.video_split = video_split
        self.label_interval = label_interval
        self.frame_between_label_num = frame_between_label_num  # the number of frame without label between two frames with label
        self.default_label_interval = default_label_interval  # default labels interval of this dataset
        if self.frame_between_label_num >= self.label_interval * self.default_label_interval:
            raise ValueError("The number of frame without label {} should be smaller than {}*{}",
                             self.frame_between_label_num, self.label_interval, self.default_label_interval)

    def _get_frame_list(self, video):
        image_path_root = os.path.join(self.image_dir, video) # , 'Imgs2_1'
        label_path_root = os.path.join(self.label_dir, video) # , 'ground-truth'
        # the list of all frame
        frame_list = sorted(glob(os.path.join(image_path_root, "*" + self.image_ext)))
        if not frame_list:
            raise FileNotFoundError(image_path_root)
        frame_id_list = [f.split("/")[-1].replace(self.image_ext, "") for f in frame_list]
        # the list of frame with labels
        label_list = sorted(glob(os.path.join(label_path_root, "*" + self.label_ext)))
        # if not label_list:
        #    raise FileNotFoundError(label_path_root)
        label_list = label_list[::self.label_interval] if self.training else label_list
        label_id_list = [f.split("/")[-1].replace(self.label_ext, "") for f in label_list]
        # the index of frames with label
        label_id_index = [frame_id_list.index(label_id) for label_id in label_id_list]

        return frame_id_list, label_id_index, image_path_root, label_path_root

    def _get_video_info(self, video):
        frame_id_list, label_id_index, image_path_root, label_path_root = self._get_frame_list(video)
        # set up video info
        video_info = []
        for image_id in frame_id_list:
            image_path = os.path.join(image_path_root, image_id + self.image_ext)
            frame_info = {'image_id': "{}/{}".format(video, image_id),
                          'image_path': image_path}
            video_info.append(frame_info)
        for index in label_id_index:
            image_id = frame_id_list[index]
            label_path = os.path.join(label_path_root, image_id + self.label_ext)
            video_info[index]['label_path'] = label_path
        return video_info, label_id_index


class VideoImageDataset(VideoDataset):
    """Video Saliency dataset class for video image
        Each index can get a frame of a video
    """

    def __init__(self, **kwargs):
        super(VideoImageDataset, self).__init__(**kwargs)
        self._set_files()

    def _set_files(self):
        self.files = []
        if self.split in list(self.video_split.keys()):
            for video in self.video_split[self.split]:
                video_info, label_id_index = self._get_video_info(video)
                self.files += [video_info[i] for i in label_id_index]
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def __getitem__(self, index):
        frame_info = self.files[index]
        return self._get_frame(frame_info)

    def _reset_files(self, label_dir):
        self.files.clear()
        self.label_dir = label_dir
        self.label_interval = 1
        self._set_files()


class VideoClipDataset(VideoDataset):
    """Video Saliency dataset class for video clip
        Each index can get a clip of a video

        [[frame_0, frame_1, ..., frame_M], <-- clip 0
         [frame_0, frame_2, ..., frame_M], <-- clip 1
         :
         [frame_0, frame_2, ..., frame_M]] <-- clip N

        Parameters
        ----------
            clip_len: int
                the number of frames of each video clip.
        Attributes
        ----------
            clips: dict
                {'video_index': video_index, 'clip_index': a list of frame index}
            flies: 2d list
                get a frame information by files[video_index][frame_index]
                e.g. files[clips['video_index]][clips['clip_index'][0]]
    """

    def __init__(self, clip_len, random_reverse_clip, **kwargs):
        super(VideoClipDataset, self).__init__(**kwargs)

        self.random_reverse_clip = random_reverse_clip
        self.clip_len = clip_len if self.frame_between_label_num == 0 else self.frame_between_label_num + 2

        self.clips = []
        self.frame_wo_label_interval = (self.label_interval * self.default_label_interval) // (
                    self.frame_between_label_num + 1)
        self._set_files()

    def _get_clips(self, video_index, label_id_index):
        indexes = []
        for index in label_id_index[:-1]:
            indexes.append(index)
            for j in range(self.frame_between_label_num):
                indexes.append(index + self.frame_wo_label_interval * (j + 1))
        indexes.append(label_id_index[-1])
        if len(indexes) < self.clip_len:
            indexes = indexes + [indexes[-1]] * (self.clip_len - len(indexes))
        clips = []
        clip_start_index = 0
        while clip_start_index <= len(indexes) - self.clip_len:
            clips.append({'video_index': video_index,
                          'clip_frame_index': indexes[clip_start_index:clip_start_index + self.clip_len]})
            clip_start_index += self.clip_len - 1 if self.training else self.clip_len
        # last clip
        if clip_start_index < len(indexes):
            clips.append(
                {'video_index': video_index, 'clip_frame_index': indexes[len(indexes) - self.clip_len:len(indexes)]})
        return clips

    def _reset_files(self, clip_len, label_dir):
        self.files.clear()
        self.clips.clear()
        self.label_dir = label_dir
        self.label_interval = 1
        self.clip_len = clip_len
        self.frame_between_label_num = 0
        self._set_files()

    def _set_files(self):
        if self.split in list(self.video_split.keys()):
            for video_index, video in enumerate(self.video_split[self.split]):
                video_info, label_id_index = self._get_video_info(video)
                if not len(video_info):
                    continue
                self.files.append(video_info)
                self.clips += self._get_clips(video_index, label_id_index)
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def __getitem__(self, index):
        clip = []
        clip_frame_index = self.clips[index]['clip_frame_index']
        video_index = self.clips[index]['video_index']
        # random reverse video when training
        if self.random_reverse_clip and random.randint(0, 1):
            clip_frame_index = clip_frame_index[::-1]
        for index, k in enumerate(clip_frame_index):
            frame_info = self.files[video_index][k]
            if index == 0:
                item, i, j, h, w, flip_index = self._get_frame(frame_info, i=0, j=0, h=0, w=0, flip_index=None, flag=False)
            else:
                item, i, j, h, w, flip_index = self._get_frame(frame_info, i=i, j=j, h=h, w=w, flip_index=flip_index, flag=True)
            clip.append(item)
        return clip

    def __len__(self):
        return len(self.clips)


def get_datasets(name_list, split_list, config_path, root, training, transforms,
                 read_clip=False, random_reverse_clip=False, label_interval=1, frame_between_label_num=0, clip_len=4):
    """
        return type of data.ConcatDataset or single dataset data.Dataset
    """
    if not isinstance(name_list, list):
        name_list = [name_list]
    if not isinstance(split_list, list):
        split_list = [split_list]
    if len(name_list) != len(split_list):
        raise ValueError("Dataset numbers must match split numbers")
    # read dataset config
    datasets_config = yaml.load(open(config_path))
    # get datasets
    dataset_list = []
    for name, split in zip(name_list, split_list):
        if name not in datasets_config.keys():
            raise ValueError("Error dataset name {}".format(name))

        dataset_config = datasets_config[name]
        dataset_config['name'] = name
        dataset_config['root'] = root
        dataset_config['split'] = split
        dataset_config['training'] = training
        dataset_config['transforms'] = transforms

        if "video_split" in dataset_config:
            dataset_config['label_interval'] = label_interval
            dataset_config['frame_between_label_num'] = frame_between_label_num
            if read_clip:
                dataset = VideoClipDataset(clip_len=clip_len,
                                           random_reverse_clip=random_reverse_clip,
                                           **dataset_config)
            else:
                dataset = VideoImageDataset(**dataset_config)
        else:
            dataset = ImageDataset(**dataset_config)

        dataset_list.append(dataset)

    if len(dataset_list) == 1:
        return dataset_list[0]
    else:
        return data.ConcatDataset(dataset_list)
