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
    def __init__(self, netG, netD, loader, optimizerD, optimizerG, checkpoint, epochs, output='./outputs', interval=50,
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

        self.fixed_noise = torch.randn(16, self.netG.nz, device=self.device)
        # 灰头发红眼睛
        t = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]
        t = t * 16
        self.fixed_c = torch.Tensor(t).to(self.device).view(16, -1)
        self.dir_output = output
        if os.path.exists(self.dir_output) is True:
            shutil.rmtree(self.dir_output)
        os.mkdir(self.dir_output)
        self.interval = interval
        self.iters = 0

        self.image_list = []

    def train(self, epoch):
        self.netD.train()
        self.netG.train()
        N = len(self.loader)
        LossD_realx, LossD_fakex, LossG_fakex = 0., 0., 0.
        local_iter = 0
        pbar = tqdm(enumerate(self.loader))
        for idx, (image, tag, fake_tag) in pbar:
            image = image.to(self.device)
            c = tag.to(self.device)
            fake_c = fake_tag.to(self.device)
            # some head
            real = image
            n = real.size(0)
            real_y = torch.full((n, 1), 1, device=self.device)
            fake_y = torch.full((n, 1), 0, device=self.device)
            noise = torch.randn(n, self.netG.nz, device=self.device)
            fake = self.netG(noise, fake_c)

            # Discriminator
            self.netD.zero_grad()
            # Real x, Matched c
            output_realistic = self.netD(real, c)
            errD_realx = self.criterion(output_realistic, real_y)

            # fake x, fake c
            output_realistic = self.netD(fake.detach(), fake_c)
            errD_fakex = self.criterion(output_realistic, fake_y)

            errD = (errD_realx + errD_fakex) / 2.
            errD.backward()
            self.optimizerD.step()

            # Generator
            self.netG.zero_grad()
            # fake x, matched c
            output_realistic = self.netD(fake, real_y)
            errG_fakex = self.criterion(output_realistic, real_y)
            errG_fakex.backward()

            self.optimizerG.step()

            # err
            errD_realx = errD_realx.item()
            errD_fakex = errD_fakex.item()
            errG_fakex = errG_fakex.item()

            pbar.set_description('[%d/%d][%d/%d]\terrD_realx: %.4f\terrD_fakex: %.4f'
                                 '\terrG_fakex: %.4f'
                  % (epoch, self.epochs, idx, N, errD_realx, errD_fakex,
                     errG_fakex))

            LossD_realx += errD_realx
            LossD_fakex += errD_fakex
            LossG_fakex += errG_fakex
            local_iter += 1
            self.iters += 1
            if self.iters % self.interval == 0:
                self.plotter.plot('Loss D', 'realx', 'Loss D', self.iters, LossD_realx / local_iter)
                self.plotter.plot('Loss D', 'fakex', 'Loss D', self.iters, LossD_fakex / local_iter)

                self.plotter.plot('Loss G', 'fakex', 'Loss G', self.iters, LossG_fakex / local_iter)
                LossD_realx, LossD_fakex, LossG_fakex = 0., 0., 0.
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

    from models.cgan import Generator, Discriminator2, weights_init
    netG = Generator(args.nz, args.ncd, args.ngf, args.nc)
    netD = Discriminator2(args.nc, args.ncd, args.ndf)
    netG = netG.to(args.device)
    netG.apply(weights_init)
    netD = netD.to(args.device)
    netD.apply(weights_init)

    from torch.optim import Adam
    optimizerD = Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
    optimizerG = Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))

    trainer = Trainer(netG, netD, loader, optimizerD, optimizerG,
                      args.checkpoint, epochs=args.epochs, output=args.output,
                      interval=args.interval, device=args.device, resume=args.resume, env=args.env)
    trainer.run()

#CUDA_VISIBLE_DEVICES=0 python train_CGAN.py --data /home/zsh_o/work/data/extra_data --checkpoint ./checkpoints/cgan.t7 --nc 3 --nz 100 --ncd 22 --lrD 5e-4 --lrG 5e-4 --batch_size 128 --epochs 200 --output ./outputs/cgan --interval 50 --env CGAN
