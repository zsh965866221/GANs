# coding = utf-8

import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

import numpy as np
from visdom import Visdom

from tqdm import tqdm
from plotter import LinePlotter
from PIL import Image

from metric import IOU


class Trainer:
    def __init__(self, netG, netD, loader_train, loader_val, optimizerD, optimizerG, checkpoint, epochs, interval=50, n_critic_D=5, n_critic_G=5,
                 device='cuda', resume=False, server='http://192.168.1.121', port=9999, env='GAN'):
        self.netG, self.netD = netG, netD
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.N_batch = len(loader_train)
        self.optimizerD, self.optimizerG = optimizerD, optimizerG
        self.checkpoint = checkpoint
        self.epochs = epochs
        self.device = torch.device(device)
        self.resume = resume
        if resume:
            if os.path.exists(checkpoint) is not True:
                raise NameError('[%s] not exist' % checkpoint)
            cp = torch.load(checkpoint)
            self.netD.load_state_dict(cp['netD'])
            self.netG.load_state_dict(cp['netG'])

        self.viz = Visdom(server=server, port=port, env=env)
        self.env = env
        self.plotter = LinePlotter(self.viz)

        self.criterion = nn.BCEWithLogitsLoss()

        self.interval = interval
        self.iters = 0

        self.image_list = []

        self.n_critic_D = n_critic_D
        self.n_critic_G = n_critic_G

        self.count_D = self.n_critic_D
        self.count_G = self.n_critic_G

        self.metric_val = IOU(n_class=self.netG.n_class)
        self.metric_train = IOU(n_class=self.netG.n_class)

    def train(self, epoch):
        self.netD.train()
        self.netG.train()
        N = len(self.loader_train)
        LossD, LossG = 0., 0.
        local_iter = 0
        pbar = tqdm(enumerate(self.loader_train))
        self.metric_train.clear()
        for idx, sample in pbar:
            # 整理batch header
            image = sample['image'].to(self.device)
            label = sample['label'].to(self.device)
            h, w = image.size(2), image.size(3)
            n = image.size(0)
            label_inv = 1 - label
            label_onehot = torch.cat([label_inv.unsqueeze(1), label.unsqueeze(1)], dim=1).float()

            real_y = torch.full((n, 1, h, w), 1, device=self.device, requires_grad=False)
            fake_y = torch.full((n, 1, h, w), 0, device=self.device, requires_grad=False)
            noise = torch.randn(n, 1, h, w, device=self.device, requires_grad=False)
            fake_label = self.netG(image, noise)
            fake_label_D = fake_label.detach()

            # Discriminator
            self.netD.zero_grad()
            # Real x, real c
            output_realistic = self.netD(image, label_onehot)
            errD_real_label_realistic = self.criterion(output_realistic, real_y)

            # Real x, fake c
            output_realistic = self.netD(image, fake_label_D)
            errD_fake_label_realistic = self.criterion(output_realistic, fake_y)

            errD = errD_real_label_realistic + errD_fake_label_realistic
            errD.backward()
            self.count_D -= 1
            if self.count_D >= 0:
                self.optimizerD.step()
                self.count_G = -10000
            elif self.count_D != -10000 - 1:
                self.count_G = self.n_critic_G

            # Generator
            self.netG.zero_grad()
            # fake x, real c G, real c D
            output_realistic = self.netD(image, fake_label)
            errG_realistic = self.criterion(output_realistic, real_y)

            errG = errG_realistic
            errG.backward()
            self.count_G -= 1
            if self.count_G >= 0:
                self.optimizerG.step()
                self.count_D = -10000
            elif self.count_G != -10000 - 1:
                self.count_D = self.n_critic_D

            # err
            errD = errD.item()
            errG = errG.item()

            pbar.set_description('[%d/%d][%d/%d]\terrD: %.4f'
                                 '\terrG: %.4f'
                  % (epoch, self.epochs, idx, N, errD, errG))

            LossD += errD
            LossG += errG
            local_iter += 1
            self.iters += 1
            if self.iters % self.interval == 0:
                self.plotter.plot('Loss', 'D', 'Loss', self.iters, LossD / local_iter)
                self.plotter.plot('Loss', 'G', 'Loss', self.iters, LossG / local_iter)
                LossD, LossG = 0., 0.
                local_iter = 0

            # metric
            output_np = fake_label.detach().cpu().numpy()
            output_np = output_np.transpose((1, 0, 2, 3)).reshape(self.netG.n_class, -1)
            predictions_np = np.argmax(output_np, axis=0)
            label_np = label.detach().cpu().numpy()
            labels_np = label_np.reshape(-1)
            self.metric_train.update(labels_np, predictions_np)

        # IOU
        mean_iou = 0.
        ious = self.metric_train.compute()
        for k, v in ious.items():
            mean_iou += v
        self.plotter.plot('IOU', 'train', 'IOU', epoch, mean_iou / len(ious))

        # 每一个epoch都保存
        state = {
            'netD': self.netD.state_dict(),
            'netG': self.netG.state_dict(),
            'epoch': epoch
        }
        torch.save(state, self.checkpoint)

    def val(self, epoch):
        self.netG.eval()
        N = len(self.loader_val)
        pbar = tqdm(enumerate(self.loader_val))
        self.metric_val.clear()
        with torch.no_grad():
            for idx, sample in pbar:
                # 整理batch header
                image = sample['image'].to(self.device)
                label = sample['label'].to(self.device)
                h, w = image.size(2), image.size(3)
                n = image.size(0)
                noise = torch.randn(n, 1, h, w, device=self.device, requires_grad=False)
                fake_label = self.netG(image, noise)
                # metric
                output_np = fake_label.detach().cpu().numpy()
                output_np = output_np.transpose((1, 0, 2, 3)).reshape(self.netG.n_class, -1)
                predictions_np = np.argmax(output_np, axis=0)
                label_np = label.detach().cpu().numpy()
                labels_np = label_np.reshape(-1)
                self.metric_val.update(labels_np, predictions_np)

        # IOU
        mean_iou = 0.
        ious = self.metric_val.compute()
        for k, v in ious.items():
            mean_iou += v
        self.plotter.plot('IOU', 'val', 'IOU', epoch, mean_iou / len(ious))

    def run(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.val(epoch)
            self.viz.save([self.env])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GAN')
    parser.add_argument('--data', required=True, type=str, help='dir data')
    parser.add_argument('--checkpoint', required=True, type=str, help='dir checkpoint')
    parser.add_argument('--interval', default=200, type=int, help='interval to show curve')
    parser.add_argument('--n_channel', default=3, type=int, help='channels of images')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--lrD', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--lrG', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--beta1', default=0.5, type=float, help='adam beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='adam beta1')
    parser.add_argument('--n_class', default=22, type=int, help='class number')
    parser.add_argument('--env', default='CGAN', type=str, help='env')
    parser.add_argument('--n_critic_D', default=5, type=int, help='number of D critic')
    parser.add_argument('--n_critic_G', default=5, type=int, help='number of G critic')
    args = parser.parse_args()

    from torch.utils.data import DataLoader
    from datasets.portrait import PortraitSegmentation, PortraitTransform

    transform = PortraitTransform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset_train = PortraitSegmentation(args.data, image_set='train', transform=transform)
    dataset_val = PortraitSegmentation(args.data, image_set='test', transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=5)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=5)

    from models.cgan_portrait import Generator, Discriminator
    netG = Generator(args.n_channel, args.n_class)
    netD = Discriminator(args.n_channel, args.n_class)
    netG = netG.to(args.device)
    netD = netD.to(args.device)

    from torch.optim import Adam
    optimizerD = Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
    optimizerG = Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))

    trainer = Trainer(netG, netD, loader_train, loader_val, optimizerD, optimizerG,
                      args.checkpoint, epochs=args.epochs,
                      interval=args.interval, device=args.device, resume=args.resume, env=args.env,
                      n_critic_D=args.n_critic_D, n_critic_G=args.n_critic_G)
    trainer.run()

#CUDA_VISIBLE_DEVICES=0 python train_CGAN.py --data /home/zsh_o/work/data/extra_data --checkpoint ./checkpoints/cgan.t7 --nc 3 --nz 100 --ncd 22 --lrD 5e-4 --lrG 5e-4 --batch_size 128 --epochs 200 --output ./outputs/cgan --interval 50 --env CGAN
