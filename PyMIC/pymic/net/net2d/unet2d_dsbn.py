# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from torch.nn.functional import interpolate
from PyMIC.pymic.net_run_dsbn.dsbn import DomainSpecificBatchNorm2d
from PyMIC.pymic.net_run_dsbn.layers import *

class MyUpsample2(nn.Module):
    def forward(self, x):
        return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)

def normalization(planes, norm='gn', num_domains=None, momentum=0.1):
    if norm == 'dsbn':
        m = DomainSpecificBatchNorm2d(planes, num_domains=num_domains, momentum=momentum)
    elif norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m

class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, num_domains=None, momentum=0.1, dropout_p = 0.5):
        super(ConvD, self).__init__()
     
        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm, num_domains, momentum=momentum)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm, num_domains, momentum=momentum)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm, num_domains, momentum=momentum)
        self.dropout_p = dropout_p
    def forward(self, x, weights=None, layer_idx=None, domain_label=None):

        # if weights == None:
        weight_1, bias_1 = self.conv1.weight, self.conv1.bias
        weight_2, bias_2 = self.conv2.weight, self.conv2.bias
        weight_3, bias_3 = self.conv3.weight, self.conv3.bias

        # else:
        #     print(weights)
        #     weight_1, bias_1 = weights[layer_idx+'.conv1.weight'], weights[layer_idx+'.conv1.bias']
        #     weight_2, bias_2 = weights[layer_idx+'.conv2.weight'], weights[layer_idx+'.conv2.bias']
        #     weight_3, bias_3 = weights[layer_idx+'.conv3.weight'], weights[layer_idx+'.conv3.bias']

        if not self.first:
            x = maxpool2D(x, kernel_size=2)

        #layer 1 conv, bn
        x = conv2d(x, weight_1, bias_1)
        if domain_label is not None:
            x, _ = self.bn1(x, domain_label)
        else:
            x = self.bn1(x)

        #layer 2 conv, bn, relu
        y = conv2d(x, weight_2, bias_2)
        if domain_label is not None:
            y, _ = self.bn2(y, domain_label)
        else:
            y = self.bn2(y)
        y = relu(y)
        y = dropout(y, self.dropout_p)
        #layer 3 conv, bn
        z = conv2d(y, weight_3, bias_3)
        if domain_label is not None:
            z, _ = self.bn3(z, domain_label)
        else:
            z = self.bn3(z)
        z = relu(z)

        return z

class ConvU(nn.Module):
    def __init__(self, planes, norm='bn', first=False, num_domains=None, momentum=0.1, dropout_p = 0.5 ):
        super(ConvU, self).__init__()
        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2*planes, planes, 3, 1, 1, bias=True)
            self.bn1   = normalization(planes, norm, num_domains, momentum=momentum)

        self.pool = MyUpsample2()
        self.conv2 = nn.Conv2d(planes, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm, num_domains, momentum=momentum)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm, num_domains, momentum=momentum)

        self.relu = nn.ReLU(inplace=True)
        self.dropout_p = dropout_p
    def forward(self, x, prev, weights=None, layer_idx=None, domain_label=None):

        if weights == None:
            if not self.first:
                weight_1, bias_1 = self.conv1.weight, self.conv1.bias
            weight_2, bias_2 = self.conv2.weight, self.conv2.bias
            weight_3, bias_3 = self.conv3.weight, self.conv3.bias

        else:
            if not self.first:
                weight_1, bias_1 = weights[layer_idx+'.conv1.weight'], weights[layer_idx+'.conv1.bias']
            weight_2, bias_2 = weights[layer_idx+'.conv2.weight'], weights[layer_idx+'.conv2.bias']
            weight_3, bias_3 = weights[layer_idx+'.conv3.weight'], weights[layer_idx+'.conv3.bias']
            
        #layer 1 conv, bn, relu
        if not self.first:
            x = conv2d(x, weight_1, bias_1, )
            if domain_label is not None:
                x, _ = self.bn1(x, domain_label)
            else:
                x = self.bn1(x)
            x = relu(x)

        #upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = conv2d(y, weight_2, bias_2, kernel_size=1, stride=1, padding=0)
        if domain_label is not None:
            y, _ = self.bn2(y, domain_label)
        else:
            y = self.bn2(y)
        y = relu(y)
        y = dropout(y, self.dropout_p)
        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        #layer 3 conv, bn
        y = conv2d(y, weight_3, bias_3)
        if domain_label is not None:
            y, _ = self.bn3(y, domain_label)
        else:
            y = self.bn3(y)
        y = relu(y)

        return y 



class UNet2D_dsbn(nn.Module):
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
        norm = self.params['norm']
        momentum = self.params['momentum']
        self.convd1 = ConvD(self.in_chns, self.ft_chns[0], norm, first=True, num_domains=self.num_domains, momentum=momentum, dropout_p = self.dropout[0])
        self.convd2 = ConvD(self.ft_chns[0], self.ft_chns[1], norm, num_domains=self.num_domains, momentum=momentum, dropout_p = self.dropout[1])
        self.convd3 = ConvD(self.ft_chns[1], self.ft_chns[2], norm, num_domains=self.num_domains, momentum=momentum, dropout_p = self.dropout[2])
        self.convd4 = ConvD(self.ft_chns[2], self.ft_chns[3], norm, num_domains=self.num_domains, momentum=momentum, dropout_p = self.dropout[3])
        self.convd5 = ConvD(self.ft_chns[3], self.ft_chns[4], norm, num_domains=self.num_domains, momentum=momentum, dropout_p = self.dropout[4])

        self.convu4 = ConvU(self.ft_chns[4], norm, first=True, num_domains=self.num_domains, momentum=momentum, dropout_p = self.dropout[3])
        self.convu3 = ConvU(self.ft_chns[3], norm, num_domains=self.num_domains, momentum=momentum, dropout_p = self.dropout[2])
        self.convu2 = ConvU(self.ft_chns[2], norm, num_domains=self.num_domains, momentum=momentum, dropout_p = self.dropout[1])
        self.convu1 = ConvU(self.ft_chns[1], norm, num_domains=self.num_domains, momentum=momentum, dropout_p = self.dropout[0])

        self.seg1 = nn.Conv2d(self.ft_chns[1], self.n_class, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, weights=None, domain_label=None):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        # if weights == None:
        x1 = self.convd1(x, domain_label=domain_label)
        x2 = self.convd2(x1, domain_label=domain_label)
        x3 = self.convd3(x2, domain_label=domain_label)
        x4 = self.convd4(x3, domain_label=domain_label)
        x5 = self.convd5(x4, domain_label=domain_label)

        y4 = self.convu4(x5, x4, domain_label=domain_label)
        y3 = self.convu3(y4, x3, domain_label=domain_label)
        y2 = self.convu2(y3, x2, domain_label=domain_label)
        y1 = self.convu1(y2, x1, domain_label=domain_label)

        y1_pred = conv2d(y1, self.seg1.weight, self.seg1.bias, kernel_size=None, stride=1, padding=0)
        # else:
        #     x1 = self.convd1(x, weights=weights, layer_idx='module.convd1', domain_label=domain_label)
        #     x2 = self.convd2(x1, weights=weights, layer_idx='module.convd2', domain_label=domain_label)
        #     x3 = self.convd3(x2, weights=weights, layer_idx='module.convd3', domain_label=domain_label)
        #     x4 = self.convd4(x3, weights=weights, layer_idx='module.convd4', domain_label=domain_label)
        #     x5 = self.convd5(x4, weights=weights, layer_idx='module.convd5', domain_label=domain_label)

        #     y4 = self.convu4(x5, x4, weights=weights, layer_idx='module.convu4', domain_label=domain_label)
        #     y3 = self.convu3(y4, x3, weights=weights, layer_idx='module.convu3', domain_label=domain_label)
        #     y2 = self.convu2(y3, x2, weights=weights, layer_idx='module.convu2', domain_label=domain_label)
        #     y1 = self.convu1(y2, x1, weights=weights, layer_idx='module.convu1', domain_label=domain_label)

        #     y1_pred = conv2d(y1, weights['module.seg1.weight'], weights['module.seg1.bias'], kernel_size=None, stride=1, padding=0)
        output = torch.sigmoid(input=y1_pred)
        if(len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
        return output

# class ConvBlock(nn.Module):
#     """
#     Two convolution layers with batch norm and leaky relu.
#     Droput is used between the two convolution layers.
    
#     :param in_channels: (int) Input channel number.
#     :param out_channels: (int) Output channel number.
#     :param dropout_p: (int) Dropout probability.
#     """
#     def __init__(self,in_channels, out_channels, dropout_p, domain_label):
#         super(ConvBlock, self).__init__()
#         self.conv_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_p),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU()
#         )
       
#     def forward(self, x):
#         return self.conv_conv(x)

# class DownBlock(nn.Module):
#     """
#     Downsampling followed by ConvBlock

#     :param in_channels: (int) Input channel number.
#     :param out_channels: (int) Output channel number.
#     :param dropout_p: (int) Dropout probability.
#     """
#     def __init__(self, in_channels, out_channels, dropout_p):
#         super(DownBlock, self).__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             ConvBlock(in_channels, out_channels, dropout_p)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)

# class UpBlock(nn.Module):
#     """
#     Upsampling followed by ConvBlock
    
#     :param in_channels1: (int) Channel number of high-level features.
#     :param in_channels2: (int) Channel number of low-level features.
#     :param out_channels: (int) Output channel number.
#     :param dropout_p: (int) Dropout probability.
#     :param bilinear: (bool) Use bilinear for up-sampling (by default).
#         If False, deconvolution is used for up-sampling. 
#     """
#     def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
#                  bilinear=True):
#         super(UpBlock, self).__init__()
#         self.bilinear = bilinear
#         if bilinear:
#             self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
#         self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

#     def forward(self, x1, x2):
#         if self.bilinear:
#             x1 = self.conv1x1(x1)
#         x1 = self.up(x1)
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class Encoder(nn.Module):
#     """
#     Encoder of 2D UNet.

#     Parameters are given in the `params` dictionary, and should include the
#     following fields:

#     :param in_chns: (int) Input channel number.
#     :param feature_chns: (list) Feature channel for each resolution level. 
#       The length should be 4 or 5, such as [16, 32, 64, 128, 256].
#     :param dropout: (list) The dropout ratio for each resolution level. 
#       The length should be the same as that of `feature_chns`.
#     """
#     def __init__(self, params):
#         super(Encoder, self).__init__()
#         self.params    = params
#         self.in_chns   = self.params['in_chns']
#         self.ft_chns   = self.params['feature_chns']
#         self.dropout   = self.params['dropout']
#         assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

#         self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
#         self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
#         self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
#         self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
#         if(len(self.ft_chns) == 5):
#             self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

#     def forward(self, x):
#         x0 = self.in_conv(x)
#         x1 = self.down1(x0)
#         x2 = self.down2(x1)
#         x3 = self.down3(x2)
#         output = [x0, x1, x2, x3]
#         if(len(self.ft_chns) == 5):
#           x4 = self.down4(x3)
#           output.append(x4)
#         return output

# class Decoder(nn.Module):
#     """
#     Decoder of 2D UNet.

#     Parameters are given in the `params` dictionary, and should include the
#     following fields:

#     :param in_chns: (int) Input channel number.
#     :param feature_chns: (list) Feature channel for each resolution level. 
#       The length should be 4 or 5, such as [16, 32, 64, 128, 256].
#     :param dropout: (list) The dropout ratio for each resolution level. 
#       The length should be the same as that of `feature_chns`.
#     :param class_num: (int) The class number for segmentation task. 
#     :param bilinear: (bool) Using bilinear for up-sampling or not. 
#         If False, deconvolution will be used for up-sampling.
#     """
#     def __init__(self, params):
#         super(Decoder, self).__init__()
#         self.params    = params
#         self.in_chns   = self.params['in_chns']
#         self.ft_chns   = self.params['feature_chns']
#         self.dropout   = self.params['dropout']
#         self.n_class   = self.params['class_num']
#         self.bilinear  = self.params['bilinear']

#         assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

#         if(len(self.ft_chns) == 5):
#             self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3], self.bilinear) 
#         self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2], self.bilinear) 
#         self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1], self.bilinear) 
#         self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0], self.bilinear) 
#         self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size = 1)

#     def forward(self, x):
#         if(len(self.ft_chns) == 5):
#             assert(len(x) == 5)
#             x0, x1, x2, x3, x4 = x 
#             x_d3 = self.up1(x4, x3)
#         else:
#             assert(len(x) == 4)
#             x0, x1, x2, x3 = x 
#             x_d3 = x3
#         x_d2 = self.up2(x_d3, x2)
#         x_d1 = self.up3(x_d2, x1)
#         x_d0 = self.up4(x_d1, x0)
#         output = self.out_conv(x_d0)
#         return output

# class UNet2D_dsbn(nn.Module):
#     """
#     An implementation of 2D U-Net.

#     * Reference: Olaf Ronneberger, Philipp Fischer, Thomas Brox:
#       U-Net: Convolutional Networks for Biomedical Image Segmentation. 
#       MICCAI (3) 2015: 234-241
    
#     Note that there are some modifications from the original paper, such as
#     the use of batch normalization, dropout, leaky relu and deep supervision.

#     Parameters are given in the `params` dictionary, and should include the
#     following fields:

#     :param in_chns: (int) Input channel number.
#     :param feature_chns: (list) Feature channel for each resolution level. 
#       The length should be 4 or 5, such as [16, 32, 64, 128, 256].
#     :param dropout: (list) The dropout ratio for each resolution level. 
#       The length should be the same as that of `feature_chns`.
#     :param class_num: (int) The class number for segmentation task. 
#     :param bilinear: (bool) Using bilinear for up-sampling or not. 
#         If False, deconvolution will be used for up-sampling.
#     :param deep_supervise: (bool) Using deep supervision for training or not.
#     """
#     def __init__(self, params):
#         super(UNet2D_dsbn, self).__init__()
#         self.params    = params
#         self.in_chns   = self.params['in_chns']
#         self.ft_chns   = self.params['feature_chns']
#         self.dropout   = self.params['dropout']
#         self.n_class   = self.params['class_num']
#         self.bilinear  = self.params['bilinear']
#         self.deep_sup  = self.params['deep_supervise']

#         assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

#         self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
#         self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
#         self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
#         self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
#         if(len(self.ft_chns) == 5):
#             self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
#             self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3], self.bilinear) 
#         self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2], self.bilinear) 
#         self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1], self.bilinear) 
#         self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0], self.bilinear) 
    
#         self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size = 1)
#         if(self.deep_sup):
#             self.out_conv1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size = 1)
#             self.out_conv2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size = 1)
#             self.out_conv3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size = 1)

#     def forward(self, x):
#         x_shape = list(x.shape)
#         if(len(x_shape) == 5):
#           [N, C, D, H, W] = x_shape
#           new_shape = [N*D, C, H, W]
#           x = torch.transpose(x, 1, 2)
#           x = torch.reshape(x, new_shape)

#         x0 = self.in_conv(x)
#         x1 = self.down1(x0)
#         x2 = self.down2(x1)
#         x3 = self.down3(x2)
#         if(len(self.ft_chns) == 5):
#           x4 = self.down4(x3)
#           x_d3 = self.up1(x4, x3)
#         else:
#           x_d3 = x3
#         x_d2 = self.up2(x_d3, x2)
#         x_d1 = self.up3(x_d2, x1)
#         x_d0 = self.up4(x_d1, x0)
#         output = self.out_conv(x_d0)
#         if(self.deep_sup):
#             out_shape = list(output.shape)[2:]
#             output1 = self.out_conv1(x_d1)
#             output1 = interpolate(output1, out_shape, mode = 'bilinear')
#             output2 = self.out_conv2(x_d2)
#             output2 = interpolate(output2, out_shape, mode = 'bilinear')
#             output3 = self.out_conv3(x_d3)
#             output3 = interpolate(output3, out_shape, mode = 'bilinear')
#             output = [output, output1, output2, output3]

#             if(len(x_shape) == 5):
#                 new_shape = [N, D] + list(output[0].shape)[1:]
#                 for i in range(len(output)):
#                     output[i] = torch.transpose(torch.reshape(output[i], new_shape), 1, 2)
#         elif(len(x_shape) == 5):
#             new_shape = [N, D] + list(output.shape)[1:]
#             output = torch.reshape(output, new_shape)
#             output = torch.transpose(output, 1, 2)

#         return output

if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'bilinear': True,
              'num_domains': 2,
              'momentum': 0.1,
              'norm': 'bn'}
    Net = UNet2D_dsbn(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 10, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)
