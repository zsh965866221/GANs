# coding = utf-8
import os
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from PIL import Image
import pandas as pd


class FacesWithTag(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = glob.glob(os.path.join(root, 'images', '*.jpg'))
        path_tags = os.path.join(root, 'tags.csv')
        self.tags = pd.read_csv(path_tags, header=None, index_col=0)
        self.hairs = ['aqua', 'gray', 'green', 'orange',
                      'red', 'white', 'black', 'blonde',
                      'blue', 'brown', 'pink', 'purple']
        self.hair_dict = {}
        for ii, hair in enumerate(self.hairs):
            self.hair_dict[hair] = ii
        self.eyes = ['aqua', 'black', 'blue', 'brown',
                     'green', 'orange', 'pink', 'purple',
                     'red', 'yellow']
        self.eye_dict = {}
        for ii, eye in enumerate(self.eyes):
            self.eye_dict[eye] = ii

        self.N_hair = len(self.hairs)
        self.N_eye = len(self.eyes)

    def one_hot(self, hair, eye):
        ihair = self.hair_dict[hair]
        ieye = self.eye_dict[eye]
        t = torch.zeros(len(self.hairs) + len(self.eyes)).float()
        t[ihair] = 1
        t[ieye + len(self.hairs)] = 1
        return t

    def get_fake(self, tag):
        t = tag.strip().split()
        hair, eye = t[0], t[2]
        ihair, ieye = self.hair_dict[hair], self.eye_dict[eye]
        fake_ihair, fake_ieye = ihair, ieye
        while (fake_ihair == ihair) and (fake_ieye == ieye):
            fake_ihair = random.randint(0, self.N_hair - 1)
            fake_ieye = random.randint(0, self.N_eye - 1)
        return '%s hair %s eyes' % (self.hairs[fake_ihair], self.eyes[fake_ieye])

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, 'images', '%d.jpg' % index)).convert('RGB')
        tag = self.tags.loc[index, 1]
        fake = self.get_fake(tag)
        if self.transform is not None:
            image = self.transform(image)
            # transform tag
            t = tag.strip().split()
            hair, eye = t[0], t[2]
            tag = self.one_hot(hair, eye)
            t = fake.strip().split()
            hair, eye = t[0], t[2]
            fake = self.one_hot(hair, eye)
        return image, tag, fake

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    transform = Compose([
        Resize(64),
        CenterCrop(64),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    root = '/home/zsh_o/work/data/extra_data'
    dataset = FacesWithTag(root=root, transform=transform)
    print(dataset[1][1])
    print(dataset[1][2])
