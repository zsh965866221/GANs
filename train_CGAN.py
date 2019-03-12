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


class Trainer:
    def __init__(self, netG, netD, loader, optimizerD, optimizerG, checkpoint, epochs, output='./outputs', interval=50, n_critic_D=5, n_critic_G=5,
                 device='cuda', resume=False, server='http://192.168.1.121', port=9999, env='GAN'):
        self.netG, self.netD = netG, netD
        self.loader = loader
        self.N_batch = len(loader)
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

        self.criterion = nn.BCELoss()
        self.weight_c_D = 1.
        self.weight_c_G = 1.

        self.fixed_noise = torch.randn(16, self.netG.nz, device=self.device)
        # 红头发红眼睛
        t = [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]
        t = t * 16
        self.fixed_c = torch.Tensor(t).to(self.device).view(16, -1)
        self.dir_output = output
        if os.path.exists(self.dir_output) is True:
            shutil.rmtree(self.dir_output)
        os.mkdir(self.dir_output)
        self.interval = interval
        self.iters = 0

        self.image_list = []

        self.n_critic_D = n_critic_D
        self.n_critic_G = n_critic_G

        self.count_D = self.n_critic_D
        self.count_G = self.n_critic_G

    def train(self, epoch):
        self.netD.train()
        self.netG.train()
        N = len(self.loader)
        LossD, LossG = 0., 0.
        local_iter = 0
        pbar = tqdm(enumerate(self.loader))
        for idx, (real_x, real_c, fake_c) in pbar:
            # 整理该batch的头
            real_x = real_x.to(self.device)
            real_c = real_c.to(self.device)
            fake_c = fake_c.to(self.device)
            n = real_x.size(0)
            real_y = torch.full((n, 1), 1, device=self.device, requires_grad=False)
            fake_y = torch.full((n, 1), 0, device=self.device, requires_grad=False)
            noise = torch.randn(n, self.netG.nz, device=self.device, requires_grad=False)
            fake_x_realc_G = self.netG(noise, real_c)
            fake_x_realc_D = fake_x_realc_G.detach()

            # Discriminator
            self.netD.zero_grad()
            # Real x, real c
            output_realistic, output_matched = self.netD(real_x, real_c)
            errD_realx_realc_realistic = self.criterion(output_realistic, real_y)
            errD_realx_realc_matched = self.criterion(output_matched, real_y)
            errD_realx_realc = errD_realx_realc_realistic + errD_realx_realc_matched * self.weight_c_D

            # Real x, fake c
            output_realistic, output_matched = self.netD(real_x, fake_c)
            errD_realx_fakec_realistic = self.criterion(output_realistic, real_y)
            errD_realx_fakec_matched = self.criterion(output_matched, fake_y)
            errD_realx_fakec = errD_realx_fakec_realistic + errD_realx_fakec_matched * self.weight_c_D

            # fake x, real c G, real c D
            output_realistic, output_matched = self.netD(fake_x_realc_D, real_c)
            errD_fakex_realc_realc_realistic = self.criterion(output_realistic, fake_y)
            errD_fakex_realc_realc_matched = self.criterion(output_matched, fake_y)
            errD_fakex_realc_realc = errD_fakex_realc_realc_realistic + errD_fakex_realc_realc_matched * self.weight_c_D

            errD = errD_realx_realc + errD_realx_fakec + errD_fakex_realc_realc
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
            output_realistic, output_matched = self.netD(fake_x_realc_G, real_c)
            errG_realc_realc_realistic = self.criterion(output_realistic, real_y)
            errG_realc_realc_matched = self.criterion(output_matched, real_y)
            errG_realc_realc = errG_realc_realc_realistic + errG_realc_realc_matched * self.weight_c_G

            errG = errG_realc_realc
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

        # 每一个epoch都保存
        state = {
            'netD': self.netD.state_dict(),
            'netG': self.netG.state_dict(),
            'epoch': epoch
        }
        torch.save(state, self.checkpoint)

    def val(self, epoch):
        self.netG.eval()
        with torch.no_grad():
            fake = self.netG(self.fixed_noise, self.fixed_c).detach().cpu()
        images = make_grid(fake, padding=2, normalize=True)
        images = images.numpy() * 255
        images = images.astype(np.uint8)
        self.viz.image(images, env=self.env, opts=dict(
            title='Output - %d' % epoch,
        ))
        img = Image.fromarray(np.transpose(images, (1, 2, 0)))
        img.save(os.path.join(self.dir_output, '%d.png' % epoch))

    def run(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.val(epoch)
            self.viz.save([self.env])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GAN')
    parser.add_argument('--data', required=True, type=str, help='dir data')
    parser.add_argument('--output', required=True, type=str, help='dir image output')
    parser.add_argument('--checkpoint', required=True, type=str, help='dir checkpoint')
    parser.add_argument('--interval', default=200, type=int, help='interval to show curve')
    parser.add_argument('--nz', default=100, type=int, help='Latent dim')
    parser.add_argument('--nc', default=3, type=int, help='channels of images')
    parser.add_argument('--ngf', default=64, type=int, help='generator feature number')
    parser.add_argument('--ndf', default=64, type=int, help='discriminator feature number')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--lrD', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--lrG', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--beta1', default=0.5, type=float, help='adam beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='adam beta1')
    parser.add_argument('--ncd', default=22, type=int, help='condition dim')
    parser.add_argument('--env', default='CGAN', type=str, help='env')
    parser.add_argument('--n_critic_D', default=5, type=int, help='number of D critic')
    parser.add_argument('--n_critic_G', default=5, type=int, help='number of G critic')
    args = parser.parse_args()

    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    from datasets.facesWithTag import FacesWithTag

    transform = Compose([
        Resize(64),
        CenterCrop(64),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = FacesWithTag(root=args.data, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

    from models.cgan import Generator, Discriminator, weights_init
    netG = Generator(args.nz, args.ncd, args.ngf, args.nc)
    netD = Discriminator(args.nc, args.ncd, args.ndf)
    netG = netG.to(args.device)
    netG.apply(weights_init)
    netD = netD.to(args.device)
    netD.apply(weights_init)

    from torch.optim import Adam
    optimizerD = Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
    optimizerG = Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))

    trainer = Trainer(netG, netD, loader, optimizerD, optimizerG,
                      args.checkpoint, epochs=args.epochs, output=args.output,
                      interval=args.interval, device=args.device, resume=args.resume, env=args.env,
                      n_critic_D=args.n_critic_D, n_critic_G=args.n_critic_G)
    trainer.run()

#CUDA_VISIBLE_DEVICES=0 python train_CGAN.py --data /home/zsh_o/work/data/extra_data --checkpoint ./checkpoints/cgan.t7 --nc 3 --nz 100 --ncd 22 --lrD 5e-4 --lrG 5e-4 --batch_size 128 --epochs 200 --output ./outputs/cgan --interval 50 --env CGAN
