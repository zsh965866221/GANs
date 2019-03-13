# coding=utf-8

import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import numpy.matlib
import os
from scipy.io import loadmat


class PortraitSegmentation(Dataset):
    def __init__(self, root, image_set='train', transform=None):
        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.transform = transform
        image_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'masks')

        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted')

        split_file = os.path.join(root, image_set.rstrip('\n') + 'list.mat')

        if not os.path.exists(split_file):
            raise ValueError('Wrong image_set entered! Please use image_set = [train, val]')
        m = loadmat(split_file)
        file_names = m['%slist' % image_set.rstrip('\n')][0]
        self.images = []
        self.masks = []
        for file in file_names:
            image = os.path.join(image_dir, '%05d.jpg' % file)
            mask = os.path.join(mask_dir, '%05d_mask.jpg' % file)
            if os.path.exists(image) is True and os.path.exists(mask) is True:
                self.images.append(image)
                self.masks.append(mask)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = img.resize((128, 128), Image.BILINEAR)
        class_mask = Image.open(self.masks[index])
        class_mask = class_mask.resize((128, 128), Image.NEAREST)

        sample = {
            'image': img,
            'label': class_mask,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_index(self, index, transform=False):
        img = Image.open(self.images[index]).convert('RGB')
        img = img.resize((128, 128), Image.BILINEAR)
        class_mask = Image.open(self.masks[index])
        class_mask = class_mask.resize((128, 128), Image.NEAREST)

        sample = {
            'image': img,
            'label': class_mask,
        }
        if transform is True:
            sample = self.transform(sample)
        return sample

    def get_test(self, path, transform=False):
        img = Image.open(path).convert('RGB')
        img = img.resize((128, 128), Image.BILINEAR)

        sample = {
            'image': img,
        }
        if transform is True:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.images)


class PortraitTransform:
    def __init__(self, mean, std):
        self.mean = np.expand_dims(np.array(mean), -1)
        self.std = np.expand_dims(np.array(std), -1)

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image_np = np.array(image, dtype=np.float).transpose((2, 0, 1))
        shape = image_np.shape
        image_np = image_np.reshape((shape[0], -1))
        image_np = (image_np - np.mean(image_np, axis=1, keepdims=True)) / np.std(image_np, axis=1, keepdims=True) * self.std + self.mean
        image_np = image_np.reshape(shape)
        image = torch.from_numpy(image_np).float()
        label = torch.from_numpy(np.array(label, dtype=np.float) / 255).long()
        sample = {
            'image': image,
            'label': label,
        }
        return sample


class PortraitTransform_test:
    def __init__(self, mean, std):
        self.mean = np.expand_dims(np.array(mean), -1)
        self.std = np.expand_dims(np.array(std), -1)

    def __call__(self, sample):
        image = sample['image']
        image_np = np.array(image, dtype=np.float).transpose((2, 0, 1))
        shape = image_np.shape
        image_np = image_np.reshape((shape[0], -1))
        image_np = (image_np - np.mean(image_np, axis=1, keepdims=True)) / np.std(image_np, axis=1, keepdims=True) * self.std + self.mean
        image_np = image_np.reshape(shape)
        image = torch.from_numpy(image_np).float()
        sample = {
            'image': image,
        }
        return sample
