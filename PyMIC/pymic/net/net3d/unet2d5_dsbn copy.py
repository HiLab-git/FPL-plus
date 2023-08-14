# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
# from PyMIC.pymic.net_run_dsbn.dsbn import DomainSpecificBatchNorm3d
# from PyMIC.pymic.net_run_dsbn.dsbn import DomainSpecificBatchNorm2d
from PyMIC.pymic.net_run_dsbn.layers import *


class MyUpsample2(nn.Module):
    def forward(self, x):
        return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)


class ConvBlockND(nn.Module):
    def __init__(self, in_channels, out_channels, num_domains=None, dim = 2, dropout_p = 0.5):
        super(ConvBlockND, self).__init__()
     
        self.conv2d_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3d_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3d_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2d1_1 = nn.BatchNorm2d(out_channels)
        self.bn2d2_1 = nn.BatchNorm2d(out_channels)
        self.bn3d1_1 = nn.BatchNorm3d(out_channels)
        self.bn3d2_1 = nn.BatchNorm3d(out_channels)
        self.bn2d1_2 = nn.BatchNorm2d(out_channels)
        self.bn2d2_2 = nn.BatchNorm2d(out_channels)
        self.bn3d1_2 = nn.BatchNorm3d(out_channels)
        self.bn3d2_2 = nn.BatchNorm3d(out_channels)
        self.dropout_p = dropout_p
        self.relu_1 = nn.PReLU()
        self.relu_2 = nn.PReLU()
        self.dim = dim
    def forward(self, x, domain_label=None):
        if (self.dim == 2):
            x = self.conv2d_1(x)
            if domain_label==0:
                x, _ = self.bn2d1_1(x)
            else:
                x, _ = self.bn2d1_2(x)
            x = self.relu_1(x)
            x = dropout(x, self.dropout_p)
            x = self.conv2d_2(x)
            if domain_label==0:
                x, _ = self.bn2d2_1(x)
            else:
                x, _ = self.bn2d2_2(x)
            x = self.relu_2(x)
        else:
            x = self.conv3d_1(x)
            if domain_label==0:
                x, _ = self.bn3d1_1(x)
            else:
                x, _ = self.bn3d1_2(x)

            x = self.relu_1(x)
            x = dropout(x, self.dropout_p)
            x = self.conv3d_2(x)
            if domain_label==0:
                x, _ = self.bn3d2_1(x)
            else:
                x, _ = self.bn3d2_2(x)
            x = self.relu_2(x)

        return x


class DownBlock(nn.Module):
    """`ConvBlockND` block followed by downsampling.

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dim: (int) Should be 2 or 3, for 2D and 3D convolution, respectively.
    :param dropout_p: (int) Dropout probability.
    :param downsample: (bool) Use downsample or not after convolution. 
    """
    def __init__(self, in_channels, out_channels, num_domains=None, 
                dim = 2, dropout_p = 0.0, downsample = True):
        super(DownBlock, self).__init__()
        self.downsample = downsample 
        self.dim = dim
        self.num_domains = num_domains
        self.conv = ConvBlockND(in_channels, out_channels, num_domains, dim, dropout_p)
        if(downsample):
            if(self.dim == 2):
                self.down_layer = nn.MaxPool2d(kernel_size = 2, stride = 2)
            else:
                self.down_layer = nn.MaxPool3d(kernel_size = 2, stride = 2)
    
    def forward(self, x, domain_label=None):
        x_shape = list(x.shape)
        if(self.dim == 2 and len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        output = self.conv(x,domain_label)
        if(self.downsample):
            output_d = self.down_layer(output)
        else:
            output_d = None 
        if(self.dim == 2 and len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
            if(self.downsample):
                new_shape = [N, D] + list(output_d.shape)[1:]
                output_d = torch.reshape(output_d, new_shape)
                output_d = torch.transpose(output_d, 1, 2)

        return output, output_d

class UpBlock(nn.Module):
    """Upsampling followed by `ConvBlockND` block
    
    :param in_channels1: (int) Input channel number for low-resolution feature map.
    :param in_channels2: (int) Input channel number for high-resolution feature map.
    :param out_channels: (int) Output channel number.
    :param dim: (int) Should be 2 or 3, for 2D and 3D convolution, respectively.
    :param dropout_p: (int) Dropout probability.
    :param bilinear: (bool) Use bilinear for up-sampling or not.
    """
    def __init__(self, in_channels1, in_channels2, out_channels, num_domains=None,
                 dim = 2, dropout_p = 0.0, bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        self.dim = dim
        self.num_domains = num_domains
        self.conv2d = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
        self.conv3d = nn.Conv3d(in_channels1, in_channels2, kernel_size=1)
        self.upsample2d = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.trans2d = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.trans3d = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
        # if bilinear:
        #     if(dim == 2):
        #         self.up = nn.Sequential(
        #             nn.Conv2d(in_channels1, in_channels2, kernel_size = 1),
        #             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        #     else:
        #         self.up = nn.Sequential(
        #             nn.Conv3d(in_channels1, in_channels2, kernel_size = 1),
        #             nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
        # else:
        #     if(dim == 2):
        #         self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        #     else:
        #         self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
            
        self.conv = ConvBlockND(in_channels2 * 2, out_channels, num_domains, dim, dropout_p)

    def forward(self, x1, x2, domain_label=None):
        x1_shape = list(x1.shape)
        x2_shape = list(x2.shape)
        if(self.dim == 2 and len(x1_shape) == 5):
            [N, C, D, H, W] = x1_shape
            new_shape = [N*D, C, H, W]
            x1 = torch.transpose(x1, 1, 2)
            x1 = torch.reshape(x1, new_shape)
            [N, C, D, H, W] = x2_shape
            new_shape = [N*D, C, H, W]
            x2 = torch.transpose(x2, 1, 2)
            x2 = torch.reshape(x2, new_shape)

        # x1 = self.up(x1)
        if self.bilinear:
            if(self.dim == 2):
                x1 = self.conv2d(x1)
                x1 = self.upsample2d(x1)
            else:
                x1 = self.conv3d(x1)
                x1 = self.upsample3d(x1)
        else:
            if(self.dim == 2):
                x1 = self.trans2d(x1)
            else:
                x1 = self.trans3d(x1)
        output = torch.cat([x2, x1], dim=1)
        output = self.conv(output,domain_label)
        if(self.dim == 2 and len(x1_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
        return output  

class UNet2D5_dsbn(nn.Module):
    """
    A 2.5D network combining 3D convolutions with 2D convolutions.

    * Reference: Guotai Wang, Jonathan Shapey, Wenqi Li, Reuben Dorent, Alex Demitriadis, 
      Sotirios Bisdas, Ian Paddick, Robert Bradford, Shaoting Zhang, Sébastien Ourselin, 
      Tom Vercauteren: Automatic Segmentation of Vestibular Schwannoma from T2-Weighted 
      MRI by Deep Spatial Attention with Hardness-Weighted Loss. 
      `MICCAI (2) 2019: 264-272. <https://link.springer.com/chapter/10.1007/978-3-030-32245-8_30>`_
    
    Note that the attention module in the orininal paper is not used here.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param conv_dims: (list) The convolution dimension (2 or 3) for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param bilinear: (bool) Using bilinear for up-sampling or not. 
        If False, deconvolution will be used for up-sampling.
    """
    def __init__(self, params):
        super(UNet2D5_dsbn, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.dims      = self.params['conv_dims']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        self.num_domains = self.params['num_domains']
        
        assert(len(self.ft_chns) == 5)

        self.block0 = DownBlock(self.in_chns, self.ft_chns[0], self.num_domains, self.dims[0], self.dropout[0], True)
        self.block1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.num_domains, self.dims[1], self.dropout[1], True)
        self.block2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.num_domains, self.dims[2], self.dropout[2], True)
        self.block3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.num_domains, self.dims[3], self.dropout[3], True)
        self.block4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.num_domains, self.dims[4], self.dropout[4], False)
        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.num_domains, 
                    self.dims[3], dropout_p = self.dropout[3], bilinear = self.bilinear) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.num_domains, 
                    self.dims[2], dropout_p = self.dropout[2], bilinear = self.bilinear) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.num_domains, 
                    self.dims[1], dropout_p = self.dropout[1], bilinear = self.bilinear) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.num_domains, 
                    self.dims[0], dropout_p = self.dropout[0], bilinear = self.bilinear) 
    
        self.out_conv = nn.Conv3d(self.ft_chns[0], self.n_class, 
            kernel_size = (1, 3, 3), padding = (0, 1, 1))

    def forward(self, x, domain_label=None):
        x0, x0_d = self.block0(x, domain_label)
        x1, x1_d = self.block1(x0_d, domain_label)
        x2, x2_d = self.block2(x1_d, domain_label)
        x3, x3_d = self.block3(x2_d, domain_label)
        x4, x4_d = self.block4(x3_d, domain_label)
        
        x = self.up1(x4, x3, domain_label)
        x = self.up2(x, x2, domain_label)
        x = self.up3(x, x1, domain_label)
        x = self.up4(x, x0, domain_label)
        output = self.out_conv(x)
        return output


if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'conv_dims': [2, 2, 3, 3, 3],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'bilinear': False,
              'num_domains':2}
    Net = UNet2D5_dsbn(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 32, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)
