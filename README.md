# GANs
GANs with PyTorch

use:
 - pytorch
 - visdom

Please install visdom for visualization

train:

```
CUDA_VISIBLE_DEVICES=0 python train_InfoGAN.py --data /home/zsh_o/work/data/extra_data --checkpoint ./checkpoints/infogan.t7 --nc 3 --nz 100 --ncd1 12 --ncd2 10 --lrD 5e-4 --lrG 5e-4 --batch_size 128 --epochs 200 --output ./outputs/infogan --interval 50 --env GAN_InfoGAN --n_critic_D 5 --n_critic_G 5
```

动画人脸数据集  
链接：https://pan.baidu.com/s/1-SRk_7D7qCH3S3lGrctnzQ   
提取码：auyt   

北化校园网可以进这个链接进行测试：http://222.199.225.62:8888/notebooks/learn/GAN/check%20InfoGAN.ipynb  
Token: zhangheng

## InfoGAN
