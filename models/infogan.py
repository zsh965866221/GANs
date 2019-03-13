# coding = utf-8
# 条件GAN

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, nz, ncd1, ncd2, ngf, nc):
        super().__init__()
        self.nz = nz
        self.ncd1 = ncd1
        self.ncd = ncd2
        self.ngf = ngf
        self.nc = nc
        self.base = nn.Sequential(
            nn.ConvTranspose2d(nz + ncd1 + ncd2, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, x, c1, c2):
        x = torch.cat([x, c1, c2], dim=1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.base(x)


class Discriminator(nn.Module):
    def __init__(self, nc, ncd1, ncd2, ndf):
        super().__init__()
        self.nc = nc
        self.ncd1 = ncd1
        self.ncd2 = ncd2
        self.ndf = ndf
        self.base = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.frealistic = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )
        self.fc1 = nn.Sequential(
            nn.Conv2d(ndf * 8, self.ncd1, 4, 1, 0, bias=False)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(ndf * 8, self.ncd2, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        x = self.base(x)
        realistic = self.frealistic(x).squeeze(-1).squeeze(-1)
        c1 = self.fc1(x).squeeze(-1).squeeze(-1)
        c2 = self.fc2(x).squeeze(-1).squeeze(-1)
        return realistic, c1, c2


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == '__main__':
    x = torch.randn(2, 3, 64, 64)
    c = torch.randn(2, 10)
    D = Discriminator(3, 10, 64)
    output = D(x, c)
    print(output)
