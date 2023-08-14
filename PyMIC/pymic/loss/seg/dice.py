# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.seg.abstract import AbstractSegLoss
from pymic.loss.seg.util import reshape_tensor_to_2D, get_classwise_dice

class DiceLoss(AbstractSegLoss):
    '''
    Dice loss for segmentation tasks.
    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    '''
    def __init__(self, params = None):
        super(DiceLoss, self).__init__(params)

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        # print(loss_input_dict.keys(),'2323')
        if loss_input_dict.get('pixel_weight', None) is not None:
            pixel_weight = loss_input_dict['pixel_weight']
            pixel_weight  = reshape_tensor_to_2D(pixel_weight) 
            if(isinstance(predict, (list, tuple))):
                predict = predict[0]
            if(self.softmax):
                predict = nn.Softmax(dim = 1)(predict)
            predict = reshape_tensor_to_2D(predict)
            soft_y  = reshape_tensor_to_2D(soft_y) 
            # print(pixel_weight.max(),pixel_weight.min(),'323233')
            dice_score = get_classwise_dice(predict, soft_y, pixel_weight)
            dice_loss  = 1.0 - dice_score.mean()
            # print('35',pixel_weight.max(),pixel_weight.min(),pixel_weight)
            return dice_loss
        # # else:
        # #     if(isinstance(predict, (list, tuple))):
        # #         predict = predict[0]
        # #     if(self.softmax):
        # #         predict = nn.Softmax(dim = 1)(predict)
        # #     predict = reshape_tensor_to_2D(predict)
        # #     soft_y  = reshape_tensor_to_2D(soft_y) 
        # #     dice_score = get_classwise_dice(predict, soft_y,pixel_weight)
        # #     dice_loss  = 1.0 - dice_score.mean()
        else:
            if(isinstance(predict, (list, tuple))):
                predict = predict[0]
            if(self.softmax):
                predict = nn.Softmax(dim = 1)(predict)
            predict = reshape_tensor_to_2D(predict)
            soft_y  = reshape_tensor_to_2D(soft_y) 
            dice_score = get_classwise_dice(predict, soft_y)
            dice_loss  = 1.0 - dice_score.mean()
            # print(dice_loss,'54')
            return dice_loss
        
class DiceLoss00(AbstractSegLoss):
    '''
    Dice loss for segmentation tasks.
    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    '''
    def __init__(self, params = None):
        super(DiceLoss, self).__init__(params)

    def forward(self, loss_input_dict):
        dice_loss = 0.0
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        # pixel_weight = loss_input_dict['pixel_weight']
        # image_weight = loss_input_dict['image_weight']
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        # print(predict.shape,soft_y.shape,pixel_weight.shape,image_weight,'29')
        print(predict.shape,'predict.shape')
        for i in range(predict.shape[0]):
            # print(predict[i:i+1].shape,soft_y[i:i+1].shape,pixel_weight.shape,image_weight,'30')
            predict_tempo = reshape_tensor_to_2D(predict[i:i+1])
            soft_y_tempo  = reshape_tensor_to_2D(soft_y[i:i+1]) 
            # pixel_weight_tempo  = reshape_tensor_to_2D(pixel_weight[i:i+1]) 
            # print(predict[i:i+1].shape,soft_y[i:i+1].shape,pixel_weight.shape,image_weight[i],image_weight.shape,'31')
            dice_score = get_classwise_dice(predict_tempo, soft_y_tempo)
            print(predict_tempo.shape,soft_y_tempo.shape,dice_score)
            # dice_loss.append((1.0 - dice_score.mean())*image_weight[i])
            dice_loss += (1.0 - dice_score.mean())# *image_weight[i]
            print(dice_score,'dice loss ')
        # print('dice_loss:',dice_loss)    
        return dice_loss/predict.shape[0]
class DiceLoss_weight(AbstractSegLoss):
    '''
    Dice loss for segmentation tasks.
    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    '''
    def __init__(self, params = None):
        super(DiceLoss_weight, self).__init__(params)

    def forward(self, loss_input_dict):
        dice_loss = 0.
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        pixel_weight = loss_input_dict['pixel_weight']
        image_weight = loss_input_dict['image_weight']
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        # print(predict.shape,soft_y.shape,pixel_weight.shape,image_weight,'29')
        for i in range(predict.shape[0]):
            # print(predict[i:i+1].shape,soft_y[i:i+1].shape,pixel_weight.shape,image_weight,'30')
            predict_tempo = reshape_tensor_to_2D(predict[i:i+1])
            soft_y_tempo  = reshape_tensor_to_2D(soft_y[i:i+1]) 
            pixel_weight_tempo  = reshape_tensor_to_2D(pixel_weight[i:i+1]) 
            
            # print(predict[i:i+1].shape,soft_y[i:i+1].shape,pixel_weight.shape,image_weight[i],image_weight.shape,'31')
            dice_score = get_classwise_dice(predict_tempo, soft_y_tempo, pixel_weight_tempo)
            # dice_loss.append((1.0 - dice_score.mean())*image_weight[i])
            dice_loss += (1.0 - dice_score.mean())*image_weight[i]
        # print(dice_loss)    
        return dice_loss/predict.shape[0]

class FocalDiceLoss(AbstractSegLoss):
    """
    Focal Dice according to the following paper:

    * Pei Wang and Albert C. S. Chung, Focal Dice Loss and Image Dilation for 
      Brain Tumor Segmentation, 2018.

    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    :param `FocalDiceLoss_beta`: (float) The hyper-parameter to set (>=1.0).
    """
    def __init__(self, params = None):
        super(FocalDiceLoss, self).__init__(params)
        self.beta = params['FocalDiceLoss_beta'.lower()] #beta should be >=1.0

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 

        dice_score = get_classwise_dice(predict, soft_y)
        dice_score = torch.pow(dice_score, 1.0 / self.beta)
        dice_loss  = 1.0 - dice_score.mean()
        return dice_loss

class NoiseRobustDiceLoss(AbstractSegLoss):
    """
    Noise-robust Dice loss according to the following paper. 
        
    * G. Wang et al. A Noise-Robust Framework for Automatic Segmentation of COVID-19 
      Pneumonia Lesions From CT Images, 
      `IEEE TMI <https://doi.org/10.1109/TMI.2020.3000314>`_, 2020.

    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    :param `NoiseRobustDiceLoss_gamma`:  (float) The hyper-parameter gammar to set (1, 2).
    """
    def __init__(self, params):
        super(NoiseRobustDiceLoss, self).__init__(params)
        self.gamma = params['NoiseRobustDiceLoss_gamma'.lower()]

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 

        numerator = torch.abs(predict - soft_y)
        numerator = torch.pow(numerator, self.gamma)
        denominator = predict + soft_y 
        numer_sum = torch.sum(numerator,  dim = 0)
        denom_sum = torch.sum(denominator,  dim = 0)
        loss_vector = numer_sum / (denom_sum + 1e-5)
        loss = torch.mean(loss_vector)   
        return loss
