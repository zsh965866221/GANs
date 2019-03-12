# coding = utf-8
# 条件GAN

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, nz, ncd, ngf, nc):
        """
        nz: 隐状态个数
        ncd: 条件状态个数
        ngf: 生成器特征基数
        nc: 生成的channel个数
        """
        super().__init__()
        self.nz = nz
        self.ncd = ncd
        self.ngf = ngf
        self.nc = nc
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + ncd, ngf * 8, 4, 1, 0, bias=False),
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

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, nc, ncd, ndf):
        super().__init__()
        self.nc = nc
        self.ncd = ncd
        self.ndf = ndf
        self.cnn1 = nn.Sequential(
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
        self.cnn2 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.embed = nn.Sequential(
            nn.Linear(ncd, 256, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.f = nn.Sequential(
            nn.Conv2d(ndf * 8 + 256, ndf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            #
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        x_mid = self.cnn1(x)
        realistic = self.cnn2(x_mid).squeeze(-1).squeeze(-1)
        c_mid = self.embed(c).unsqueeze(-1).unsqueeze(-1)
        c_fill = c_mid.repeat(1, 1, 4, 4)
        x = torch.cat([x_mid, c_fill], dim=1)
        matched = self.f(x).squeeze(-1).squeeze(-1)
        return realistic, matched


class Discriminator2(nn.Module):
    def __init__(self, nc, ncd, ndf):
        super().__init__()
        self.nc = nc
        self.ncd = ncd
        self.ndf = ndf
        self.cnn1 = nn.Sequential(
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
        self.embed = nn.Sequential(
            nn.Linear(ncd, 256, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.f = nn.Sequential(
            nn.Conv2d(ndf * 8 + 256, ndf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            #
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        x_mid = self.cnn1(x)
        c_mid = self.embed(c).unsqueeze(-1).unsqueeze(-1)
        c_fill = c_mid.repeat(1, 1, 4, 4)
        x = torch.cat([x_mid, c_fill], dim=1)
        realistic = self.f(x)
        return realistic.squeeze(-1).squeeze(-1)


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
