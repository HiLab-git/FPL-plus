# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.net_run_ssl.ssl_abstract import SSLSegAgent
from pymic.util.ramps import get_rampup_ratio

class SSLURPC(SSLSegAgent):
    """
    Uncertainty-Rectified Pyramid Consistency for semi-supervised segmentation.
    
    * Reference: Xiangde Luo, Guotai Wang*, Wenjun Liao, Jieneng Chen, Tao Song, Yinan Chen, 
      Shichuan Zhang, Dimitris N. Metaxas, Shaoting Zhang. 
      Semi-Supervised Medical Image Segmentation via Uncertainty Rectified Pyramid Consistency .
      `Medical Image Analysis 2022. <https://doi.org/10.1016/j.media.2022.102517>`_
    
    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `semi_supervised_learning` is needed. See :doc:`usage.ssl` for details.
    """
    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        ssl_cfg     = self.config['semi_supervised_learning']
        iter_max     = self.config['training']['iter_max']
        rampup_start = ssl_cfg.get('rampup_start', 0)
        rampup_end   = ssl_cfg.get('rampup_end', iter_max)
        train_loss  = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        self.net.train()
        kl_distance = nn.KLDivLoss(reduction='none')
        for it in range(iter_valid):
            try:
                data_lab = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data_lab = next(self.trainIter)
            try:
                data_unlab = next(self.trainIter_unlab)
            except StopIteration:
                self.trainIter_unlab = iter(self.train_loader_unlab)
                data_unlab = next(self.trainIter_unlab)

            # get the inputs
            x0   = self.convert_tensor_type(data_lab['image'])
            y0   = self.convert_tensor_type(data_lab['label_prob'])  
            x1   = self.convert_tensor_type(data_unlab['image'])
            inputs = torch.cat([x0, x1], dim = 0)               
            inputs, y0 = inputs.to(self.device), y0.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward pass
            outputs_list = self.net(inputs)
            n0 = list(x0.shape)[0] 

            # get supervised loss
            p0 = [output_i[:n0] for output_i in outputs_list]
            loss_sup = self.get_loss_value(data_lab, p0, y0)

            # get average probability across scales
            outputs_soft_list = [torch.softmax(item, dim=1) for item in outputs_list]
            outputs_soft_avg  = torch.mean(torch.stack(outputs_soft_list),dim = 0)
            p1_avg = outputs_soft_avg[n0:] * 0.99 + 0.005 # for unannotated images

            # regularization loss
            loss_reg = 0.0
            for soft_i in outputs_soft_list:
                p1_i = soft_i[n0:] * 0.99 + 0.005
                var  = torch.sum(kl_distance(
                        torch.log(p1_i), p1_avg), dim=1, keepdim=True)
                exp_var = torch.exp(-var)            
                square_e= torch.square(p1_avg - p1_i)
                loss_i  = torch.mean(square_e * exp_var)  / \
                            (torch.mean(exp_var) + 1e-8) + torch.mean(var)
                loss_reg += loss_i
            loss_reg = loss_reg / len(outputs_list)
                        
            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end, "sigmoid")
            regular_w = ssl_cfg.get('regularize_w', 0.1) * rampup_ratio
            loss = loss_sup + regular_w*loss_reg

            loss.backward()
            self.optimizer.step()
            if(self.scheduler is not None and \
                not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)):
                self.scheduler.step()
            train_loss = train_loss + loss.item()
            train_loss_sup = train_loss_sup + loss_sup.item()
            train_loss_reg = train_loss_reg + loss_reg.item() 
            # get dice evaluation for each class in annotated images
            if(isinstance(p0, tuple) or isinstance(p0, list)):
                p0 = p0[0] 
            p0_argmax = torch.argmax(p0, dim = 1, keepdim = True)
            p0_soft   = get_soft_label(p0_argmax, class_num, self.tensor_type)
            p0_soft, y0 = reshape_prediction_and_ground_truth(p0_soft, y0) 
            dice_list   = get_classwise_dice(p0_soft, y0)
            train_dice_list.append(dice_list.cpu().numpy())
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
        train_avg_dice = train_cls_dice.mean()

        train_scalers = {'loss': train_avg_loss, 'loss_sup':train_avg_loss_sup,
            'loss_reg':train_avg_loss_reg, 'regular_w':regular_w,
            'avg_dice':train_avg_dice,     'class_dice': train_cls_dice}
        return train_scalers
