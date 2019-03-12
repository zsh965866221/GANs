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
        Loss_D, Loss_G, Loss_D_x, Loss_D_G_z1, Loss_D_G_z2 = 0., 0., 0., 0., 0.
        local_iter = 0
        pbar = tqdm(enumerate(self.loader))
        for idx, image in pbar:
            image = image.to(self.device)
            # some header
            real_x = image
            n = real_x.size(0)
            real_y = torch.full((n, 1), 1, device=self.device, requires_grad=False)
            fake_y = torch.full((n, 1), 0, device=self.device, requires_grad=False)
            noise = torch.randn(n, self.netG.nz, device=self.device, requires_grad=False)
            fake_x = self.netG(noise)

            # Discriminator
            self.netD.zero_grad()
            # Real
            output = self.netD(real_x)
            errD_real = self.criterion(output, real_y)
            D_x = output.mean().item()
            # fake
            output = self.netD(fake_x.detach())
            errD_fake = self.criterion(output, fake_y)
            D_G_z1 = output.mean().item()

            errD = (errD_real + errD_fake) / 2.
            errD.backward()
            self.optimizerD.step()

            # generator
            self.netG.zero_grad()
            output = self.netD(fake_x)
            errG = self.criterion(output, real_y)
            D_G_z2 = output.mean().item()

            errG.backward()
            self.optimizerG.step()

            pbar.set_description('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, self.epochs, idx, N, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            Loss_D += errD.item()
            Loss_G += errG.item()
            Loss_D_x += D_x
            Loss_D_G_z1 += D_G_z1
            Loss_D_G_z2 += D_G_z2
            local_iter += 1
            self.iters += 1
            if self.iters % self.interval == 0:
                self.plotter.plot('Loss', 'D', 'Loss', self.iters, Loss_D / local_iter)
                self.plotter.plot('Loss', 'G', 'Loss', self.iters, Loss_G / local_iter)

                self.plotter.plot('Error', 'real', 'Error', self.iters, Loss_D_x / local_iter)
                self.plotter.plot('Error', 'Before', 'Error', self.iters, Loss_D_G_z1 / local_iter)
                self.plotter.plot('Error', 'After', 'Error', self.iters, Loss_D_G_z2 / local_iter)
                Loss_D, Loss_G, Loss_D_x, Loss_D_z1, Loss_D_z2 = 0., 0., 0., 0., 0.
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
            fake = self.netG(self.fixed_noise).detach().cpu()
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
    parser.add_argument('--env', default='CGAN', type=str, help='env')
    args = parser.parse_args()

    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    from datasets.faces import Faces

    transform = Compose([
        Resize(64),
        CenterCrop(64),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Faces(root=args.data, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

    from models.gan import Generator, Discriminator, weights_init
    netG = Generator(args.nz, args.ngf, args.nc)
    netD = Discriminator(args.nc, args.ndf)
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

#CUDA_VISIBLE_DEVICES=0 python train_GAN.py --data /home/zsh_o/work/data/faces --checkpoint ./checkpoints/gan.t7 --nc 3 --nz 100 --lrD 5e-4 --lrG 5e-4 --batch_size 128 --epochs 200 --output ./outputs/gan --interval 50
