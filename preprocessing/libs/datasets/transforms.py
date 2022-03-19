
import random

from PIL import Image, ImageOps
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils import data
import math
import numbers
import numpy as np


class Video_train_Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, i, j, h, w, flip_index, flag):
        for t in self.transforms:
            img, i, j, h, w, flip_index = t(img, i, j, h, w, flip_index, flag)
        return img, i, j, h, w, flip_index

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def get_transforms(image_mode, input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    data_transforms = {
        'train': Video_train_Compose([
            # ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.3, image_mode=image_mode),
            RandomResizedCrop(input_size, image_mode),
            RandomFlip(image_mode),
            ToTensor(),
            Normalize(mean=mean,
                      std=std)
        ]) if image_mode else Video_train_Compose([
            FixedResize(size=input_size),
            ToTensor(),
            Normalize(mean=mean,
                      std=std)
        ]),
        'val': Video_train_Compose([
            FixedResize(size=input_size),
            ToTensor(),
            Normalize(mean=mean,
                      std=std)
            # ToTensor()
        ]),
        'test': Video_train_Compose([
            FixedResize(size=input_size),
            ToTensor(),
            Normalize(mean=mean,
                      std=std)
            # ToTensor()
        ]),
    }
    return data_transforms


class ColorJitter(transforms.ColorJitter):
    def __init__(self, image_mode, **kwargs):
        super(ColorJitter, self).__init__(**kwargs)
        self.transform = None
        self.image_mode = image_mode

    def __call__(self, sample):
        if self.transform is None or self.image_mode:
            self.transform = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
        sample['image'] = self.transform(sample['image'])
        return sample


class RandomResizedCrop(object):
    """
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, size, image_mode, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.i, self.j, self.h, self.w = None, None, None, None
        self.image_mode = image_mode

    def __call__(self, sample,  i=0, j=0, h=0, w=0, flip_index=None, flag=False):
        image, label = sample['image'], sample['label']
        if not flag:
            if self.i is None or self.image_mode:
                self.i, self.j, self.h, self.w = transforms.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        else:
            self.i, self.j, self.h, self.w = i, j, h, w
        image = F.resized_crop(image, self.i, self.j, self.h, self.w, self.size, Image.BILINEAR)
        label = F.resized_crop(label, self.i, self.j, self.h, self.w, self.size, Image.BILINEAR)
        i, j, h, w = self.i, self.j, self.h, self.w
        sample['image'], sample['label'] = image, label
        return sample, i, j, h, w, flip_index


class RandomFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    """
    def __init__(self, image_mode):
        self.rand_flip_index = None
        self.image_mode = image_mode

    def __call__(self, sample, i, j, h, w, flip_index, flag):
        image, label = sample['image'], sample['label']
        if not flag:
            if self.rand_flip_index is None or self.image_mode:
                self.rand_flip_index = random.randint(-1,2)
        else:
            self.rand_flip_index = flip_index
        # 0: horizontal flip, 1: vertical flip, -1: horizontal and vertical flip
        if self.rand_flip_index == 0:
            image = F.hflip(image)
            label = F.hflip(label)
        elif self.rand_flip_index == 1:
            image = F.vflip(image)
            label = F.vflip(label)
        elif self.rand_flip_index == 2:
            image = F.vflip(F.hflip(image))
            label = F.vflip(F.hflip(label))
        flip_index = self.rand_flip_index
        sample['image'], sample['label'] = image, label
        return sample, i, j, h, w, flip_index


# class FixedResize(object):
#     def __init__(self, size):
#         self.size = tuple(reversed(size))  # size: (h, w)
#
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#
#         assert image.size == label.size
#
#         image = image.resize(self.size, Image.BILINEAR)
#         label = label.resize(self.size, Image.NEAREST)
#
#         return {'image': image,
#                 'label': label}


class FixedResize(object):
    """ Resize PIL image use both for training and inference"""
    def __init__(self, size):
        self.size = size

    def __call__(self, sample, i, j, h, w, flip_index, flag):
        image, label = sample['image'], sample['label']
        image = F.resize(image, self.size, Image.BILINEAR)
        if label is not None:
            label = F.resize(label, self.size, Image.BILINEAR)
        sample['image'], sample['label'] = image, label
        return sample, i, j, h, w, flip_index


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number): 
            self.size = (int(size), int(size))
        else:
            self.size = size  # h, w
        self.padding = padding

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.padding > 0:
            image = ImageOps.expand(image, border=self.padding, fill=0)
            label = ImageOps.expand(label, border=self.padding, fill=0)

        assert label.size == label.size
        w, h = image.size
        th, tw = self.size
        if w == tw and h == th:
            return {'image': image,
                    'label': label}
        if w < tw or h < th:
            image = image.resize((tw, th), Image.BILINEAR)
            label = label.resize((tw, th), Image.NEAREST)

            return {'image': image,
                    'label': label}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        image = image.crop((x1, y1, x1 + tw, y1 + th))
        label = label.crop((x1, y1, x1 + tw, y1 + th))
        sample['image'], sample['label'] = image, label
        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        sample['image'], sample['label'] = image, label
        return sample


# class Normalize(object):
#     """Normalize a tensor image with mean and standard deviation.
#     Args:
#         mean (tuple): means for each channel.
#         std (tuple): standard deviations for each channel.
#     """
#     def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, sample):
#         image = np.array(sample['image']).astype(np.float32)
#         label = np.array(sample['label']).astype(np.float32)
#         image /= 255.0
#         image -= self.mean
#         image /= self.std
#
#         return {'image': image,
#                 'label': label}


class Normalize(object):
    """ Normalize a tensor image with mean and standard deviation.
        args:    tensor (Tensor) – Tensor image of size (C, H, W) to be normalized.
        Returns: Normalized Tensor image.
    """
    # default caffe mode
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample, i, j, h, w, flip_index, flag):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        sample['image'], sample['label'] = image, label
        return sample, i, j, h, w, flip_index


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
#         label = np.expand_dims(np.array(sample['label']).astype(np.float32), -1).transpose((2, 0, 1))
#         label[label == 255] = 0
#
#         image = torch.from_numpy(image).float()
#         label = torch.from_numpy(label).float()
#
#         return {'image': image,
#                 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, i, j, h, w, flip_index, flag):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # Image range from [0~255] to [0.0 ~ 1.0]
        image = F.to_tensor(image)
        if label is not None:
            label = torch.from_numpy(np.array(label)).unsqueeze(0).float()
        sample['image'], sample['label'] = image, label
        return sample, i, j, h, w, flip_index
