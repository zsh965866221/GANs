# coding = utf-8
# CGAN for segmentation, use FCN8s for base net, resize to (128, 128)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .drn import DRN


class Generator(nn.Module):
    def __init__(self, n_channel, n_class, ngf=32):
        super().__init__()
        self.n_channel = n_channel
        self.n_class = n_class
        self.ngd = ngf
        self.main = nn.Sequential(
            nn.Conv2d(n_channel, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf, ngf * 2, 3, 1, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf * 2, ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf * 4, ngf * 8, 3, 1, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf * 8, n_class, 3, 1, 1)
        )

    def forward(self, x, c=None):
        y = self.main(x)
        return y


class Discriminator(nn.Module):
    def __init__(self, n_channel, n_class, ndf=32):
        super().__init__()
        self.n_channel = n_channel
        self.n_class = n_class
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv2d(n_channel + n_class, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),

            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(ndf * 8, 1, 3, 1, 1)
        )

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        y = self.main(x)
        return y
