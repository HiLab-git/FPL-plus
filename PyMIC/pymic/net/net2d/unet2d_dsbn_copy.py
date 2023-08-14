# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from torch.nn.functional import interpolate
from PyMIC.pymic.net_run_dsbn.layers import *


class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, out_channels, num_domains):
        super(_DomainSpecificBatchNorm, self).__init__()
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(out_channels) for _ in range(num_domains)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label[0]]
        return bn(x)


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))



class ConvBlock(nn.Module):
    """
    Two convolution layers with batch norm and leaky relu.
    Droput is used between the two convolution layers.
    
    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self,in_channels, out_channels, dropout_p, num_domains):
        super(ConvBlock, self).__init__()
        self.Conv2d_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.Conv2d_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.m = DomainSpecificBatchNorm2d(out_channels,num_domains=num_domains)
        # self.conv_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(out_channels),
        #     m(domain_label=domain_label),
        #     nn.LeakyReLU(),
        #     nn.Dropout(dropout_p),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(out_channels),
        #     m(domain_label=domain_label),
        #     nn.LeakyReLU()
        # )
       
    def forward(self, x, domain_label):
        
        x = self.Conv2d_1(x)
        x = self.m(x,domain_label)
        x = self.relu(x)
     
        x = self.dropout(x)
    
        x = self.Conv2d_2(x)
        x = self.m(x,domain_label)
        x = self.relu(x)
        return x

class DownBlock(nn.Module):
    """
    Downsampling followed by ConvBlock

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p, num_domains):
        super(DownBlock, self).__init__()
        self.MaxPool2d = nn.MaxPool2d(2)
        self.ConvBlock = ConvBlock(in_channels, out_channels, dropout_p, num_domains)
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     ConvBlock(in_channels, out_channels, dropout_p, num_domains)
        # )

    def forward(self, x, domain_label):
        x = self.MaxPool2d(x)
        x = self.ConvBlock(x,domain_label)
        return x

class UpBlock(nn.Module):
    """
    Upsampling followed by ConvBlock
    
    :param in_channels1: (int) Channel number of high-level features.
    :param in_channels2: (int) Channel number of low-level features.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    :param bilinear: (bool) Use bilinear for up-sampling (by default).
        If False, deconvolution is used for up-sampling. 
    """
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, num_domains,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p,num_domains)

    def forward(self, x1, x2,domain_label):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x,domain_label)

class Encoder(nn.Module):
    """
    Encoder of 2D UNet.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    """
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.num_domains = self.params['num_domains']
        
        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0],self.num_domains)
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1],self.num_domains)
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2],self.num_domains)
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3],self.num_domains)
        if(len(self.ft_chns) == 5):
            self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4],self.num_domains)

    def forward(self, x, domain_label):
        x0 = self.in_conv(x,domain_label)
        x1 = self.down1(x0,domain_label)
        x2 = self.down2(x1,domain_label)
        x3 = self.down3(x2,domain_label)
        output = [x0, x1, x2, x3]
        if(len(self.ft_chns) == 5):
          x4 = self.down4(x3,domain_label)
          output.append(x4)
        return output

class Decoder(nn.Module):
    """
    Decoder of 2D UNet.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param bilinear: (bool) Using bilinear for up-sampling or not. 
        If False, deconvolution will be used for up-sampling.
    """
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        self.num_domains = self.params['num_domains']
        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        if(len(self.ft_chns) == 5):
            self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3],self.num_domains, self.bilinear) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2],self.num_domains, self.bilinear) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1],self.num_domains, self.bilinear) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0],self.num_domains, self.bilinear) 
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size = 1)

    def forward(self, x,domain_label):
        if(len(self.ft_chns) == 5):
            assert(len(x) == 5)
            x0, x1, x2, x3, x4 = x 
            x_d3 = self.up1(x4, x3,domain_label)
        else:
            assert(len(x) == 4)
            x0, x1, x2, x3 = x 
            x_d3 = x3
        x_d2 = self.up2(x_d3, x2,domain_label)
        x_d1 = self.up3(x_d2, x1,domain_label)
        x_d0 = self.up4(x_d1, x0,domain_label)
        output = self.out_conv(x_d0)
        return output


class UNet2D_dsbn(nn.Module):
    """
    An implementation of 2D U-Net.

    * Reference: Olaf Ronneberger, Philipp Fischer, Thomas Brox:
      U-Net: Convolutional Networks for Biomedical Image Segmentation. 
      MICCAI (3) 2015: 234-241
    
    Note that there are some modifications from the original paper, such as
    the use of batch normalization, dropout, leaky relu and deep supervision.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param bilinear: (bool) Using bilinear for up-sampling or not. 
        If False, deconvolution will be used for up-sampling.
    :param deep_supervise: (bool) Using deep supervision for training or not.
    """
    def __init__(self, params):
        super(UNet2D_dsbn, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        self.deep_sup  = self.params['deep_supervise']
        self.num_domains = self.params['num_domains']
        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0],self.num_domains)
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1],self.num_domains)
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2],self.num_domains)
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3],self.num_domains)
        if(len(self.ft_chns) == 5):
            self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4],self.num_domains)
            self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3],self.num_domains, self.bilinear) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2],self.num_domains, self.bilinear) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1],self.num_domains, self.bilinear) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0],self.num_domains, self.bilinear) 
    
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size = 1)
        if(self.deep_sup):
            self.out_conv1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size = 1)
            self.out_conv2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size = 1)
            self.out_conv3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size = 1)

    def forward(self, x, domain_label):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)

        x0 = self.in_conv(x,domain_label)
        x1 = self.down1(x0,domain_label)
        x2 = self.down2(x1,domain_label)
        x3 = self.down3(x2,domain_label)
        if(len(self.ft_chns) == 5):
          x4 = self.down4(x3,domain_label)
          x_d3 = self.up1(x4, x3, domain_label)
        else:
          x_d3 = x3
        x_d2 = self.up2(x_d3, x2, domain_label)
        x_d1 = self.up3(x_d2, x1, domain_label)
        x_d0 = self.up4(x_d1, x0, domain_label)
        output = self.out_conv(x_d0)
        if(self.deep_sup):
            out_shape = list(output.shape)[2:]
            output1 = self.out_conv1(x_d1)
            output1 = interpolate(output1, out_shape, mode = 'bilinear')
            output2 = self.out_conv2(x_d2)
            output2 = interpolate(output2, out_shape, mode = 'bilinear')
            output3 = self.out_conv3(x_d3)
            output3 = interpolate(output3, out_shape, mode = 'bilinear')
            output = [output, output1, output2, output3]

            if(len(x_shape) == 5):
                new_shape = [N, D] + list(output[0].shape)[1:]
                for i in range(len(output)):
                    output[i] = torch.transpose(torch.reshape(output[i], new_shape), 1, 2)
        elif(len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)

        return output

if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'bilinear': True,
              'num_domains': 2,
              'momentum': 0.1,
              'norm': 'bn',
              'num_domains':2}
    Net = UNet2D_dsbn(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 10, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)
