# coding = utf-8
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from PIL import Image


class Faces(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = glob.glob(os.path.join(root, '*.jpg'))

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

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
    root = '/home/zsh_o/work/data/faces/'
    dataset = Faces(root=root, transform=transform)
    print(dataset[0])
