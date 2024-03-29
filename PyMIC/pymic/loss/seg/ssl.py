
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from pymic.loss.seg.util import reshape_tensor_to_2D

class EntropyLoss(nn.Module):
    """
    Entropy Minimization for segmentation tasks.
    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    """
    def __init__(self, params = None):
        super(EntropyLoss, self).__init__(params)
        
    def forward(self, loss_input_dict):
        """
        Forward pass for calculating the loss.
        The arguments should be written in the `loss_input_dict` dictionary, 
        and it has the following fields:

        :param `prediction`: (tensor) Prediction of a network, with the 
            shape of [N, C, D, H, W] or [N, C, H, W].

        :return: Loss function value.
        """
        predict = loss_input_dict['prediction']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)

        # for numeric stability
        predict = predict * 0.999 + 5e-4
        C = list(predict.shape)[1]
        entropy = torch.sum(-predict*torch.log(predict), dim=1) / np.log(C)
        avg_ent = torch.mean(entropy)
        return avg_ent
    
class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for segmentation tasks.
    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    """
    def __init__(self, params = None):
        super(TotalVariationLoss, self).__init__(params)
        
    def forward(self, loss_input_dict):
        """
        Forward pass for calculating the loss.
        The arguments should be written in the `loss_input_dict` dictionary, 
        and it has the following fields:

        :param `prediction`: (tensor) Prediction of a network, with the 
            shape of [N, C, D, H, W] or [N, C, H, W].

        :return: Loss function value.
        """
        predict = loss_input_dict['prediction']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)

        # for numeric stability
        predict = predict * 0.999 + 5e-4
        dim = list(predict.shape)[2:]
        if(dim == 2):
            pred_min = -1 * nn.functional.max_pool2d(-1*predict, (3, 3), 1, 1)
            pred_max = nn.functional.max_pool2d(pred_min, (3, 3), 1, 1)
        else:
            pred_min = -1 * nn.functional.max_pool3d(-1*predict, (3, 3, 3), 1, 1)
            pred_max = nn.functional.max_pool3d(pred_min, (3, 3, 3), 1, 1)            
        contour = torch.relu(pred_max - pred_min)
        length  = torch.mean(contour)
        return length