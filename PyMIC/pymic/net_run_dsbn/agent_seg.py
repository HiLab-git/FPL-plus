# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy
import os
import time
import logging
import scipy
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from datetime import datetime
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.io.nifty_dataset import NiftyDataset,NiftyDataset_npy
from pymic.net.net_dict_seg import SegNetDict
from pymic.net_run_dsbn.agent_abstract import NetRunAgent
from pymic.net_run_dsbn.infer_func import Inferer
from pymic.loss.loss_dict_seg import SegLossDict
from pymic.loss.seg.combined import CombinedLoss
from pymic.loss.seg.deep_sup import DeepSuperviseLoss
from pymic.loss.seg.util import get_soft_label,dice_weight_loss
from pymic.loss.seg.util import reshape_prediction_and_ground_truth,reshape_tensor_to_2D
from pymic.loss.seg.util import get_classwise_dice
from pymic.transform.trans_dict import TransformDict
from pymic.util.post_process import PostProcessDict
from pymic.util.image_process import convert_label
from pymic.net.net3d.unet2d5_dsbn import init_weights
from pymic.util.make_noise import make_noise_masks_3d,make_noise_masks_2d
# from script.nonlinear import nonlinear_transformation
class SegmentationAgent(NetRunAgent):
    def __init__(self, config, stage = 'train'):
        super(SegmentationAgent, self).__init__(config, stage)
        self.transform_dict   = TransformDict
        self.net_dict         = SegNetDict
        self.postprocess_dict = PostProcessDict
        self.postprocessor    = None
        
    def get_stage_dataset_from_config(self, stage):
        assert(stage in ['1_train', '1_valid', '1_test', '2_train', '2_valid', '2_test', 'test'])
        root_dir  = self.config['dataset']['root_dir']
        modal_num = self.config['dataset'].get('modal_num', 1)
        real_stage = stage.split('_')[-1]
        transform_key = real_stage +  '_transform'
        if(real_stage == "valid" and transform_key not in self.config['dataset']):
            transform_key = "train_transform"
        transform_names = self.config['dataset'][transform_key]
        
        self.transform_list  = []
        if(transform_names is None or len(transform_names) == 0):
            data_transform = None 
        else:
            transform_param = self.config['dataset']
            transform_param['task'] = 'segmentation' 
            for name in transform_names:
                if(name not in self.transform_dict):
                    raise(ValueError("Undefined transform {0:}".format(name))) 
                one_transform = self.transform_dict[name](transform_param)
                self.transform_list.append(one_transform)
            data_transform = transforms.Compose(self.transform_list)

        csv_file = self.config['dataset'].get(stage + '_csv', None)
        self.train_fpl_uda = self.config['training']['train_fpl_uda']
        if self.train_fpl_uda:
            dataset  = NiftyDataset(root_dir  = root_dir,
                                    csv_file  = csv_file,
                                    modal_num = modal_num,
                                    with_label= not (stage == 'test'),
                                    transform = data_transform
                                    )
        else:
            dataset  = NiftyDataset(root_dir  = root_dir,
                                    csv_file  = csv_file,
                                    modal_num = modal_num,
                                    with_label= not (stage == 'test'),
                                    transform = data_transform )
        return dataset

    def create_network(self):
        if(self.net is None):
            net_name = self.config['network']['net_type']
            print(net_name)
            if(net_name not in self.net_dict):
                raise ValueError("Undefined network {0:}".format(net_name))
            self.net = self.net_dict[net_name](self.config['network'])
            
        if(self.tensor_type == 'float'):
            self.net.float()
        else:
            self.net.double()
        

        if self.config['training']['dis']  == True:
                from PyMIC.pymic.net.net3d.unet2d5_dsbn import Dis
                self.disseg = Dis(self.config['network']['class_num'])
                if self.config['training']['dis_para'] is not None:
                    self.disseg.load_state_dict(torch.load(self.config['training']['dis_para'],map_location='cpu'))
                param_number_dis = sum(p.numel() for p in self.disseg.parameters() if p.requires_grad)
                logging.info('parameter number of disc {0:}'.format(param_number_dis))
        param_number = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        
        logging.info('parameter number {0:}'.format(param_number))
        

    def get_parameters_to_update(self):
        if self.config['training']['dis']  == True:
            self.disSeg_opt = torch.optim.Adam(self.disseg.parameters(),lr=0.0001,betas=(0.5,0.999))
        return self.net.parameters()

    def create_loss_calculator(self):
        if(self.loss_dict is None):
            self.loss_dict = SegLossDict
        loss_name = self.config['training']['loss_type']
        if isinstance(loss_name, (list, tuple)):
            base_loss = CombinedLoss(self.config['training'], self.loss_dict)
            
        elif (loss_name not in self.loss_dict):
            raise ValueError("Undefined loss function {0:}".format(loss_name))
            
        else:
            base_loss = self.loss_dict[loss_name](self.config['training'])
            
        if(self.config['network'].get('deep_supervise', False)):
            weight = self.config['network'].get('deep_supervise_weight', None)
            params = {'deep_supervise_weight': weight, 'base_loss':base_loss}
            self.loss_calculator = DeepSuperviseLoss(params)
            
        else:
            self.loss_calculator = base_loss
                
    def get_loss_value(self, data, pred, gt, fpl_uda = False):
        loss_input_dict = {'prediction':pred, 'ground_truth': gt}
        if fpl_uda:
            if data.get('pixel_weight', None) is not None:
                loss_input_dict['pixel_weight'] = data['pixel_weight'].to(pred.device)
                if data.get('image_weight', None) is not None:
                    loss_input_dict['image_weight'] = data['image_weight'].to(pred.device)
        loss_value = self.loss_calculator(loss_input_dict)
        return loss_value
    
    def set_postprocessor(self, postprocessor):
        """
        Set post processor after prediction. 

        :param postprocessor: post processor, such as an instance of 
            `pymic.util.post_process.PostProcess`.
        """
        self.postprocessor = postprocessor
    def repeat_dataloader(sef,iterable):
        while True:
            for x in iterable:
                yield x

    def training_dual_doamian(self):
        import tqdm
        
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        number_domians = self.config['network']['num_domains']
        train_loss  = 0
        train_dice_list_0 = []
        train_dice_list_1 = []
        self.net.train()
        self.criterionL2 = nn.MSELoss()
        if number_domians == 2:
            trainIter_0  = iter(self.train_loader_1)
            trainIter_1  = iter(self.train_loader_2)
        elif number_domians == 1:
            trainIter_0  = iter(self.train_loader_1)
        for it in range(iter_valid):
            if number_domians == 2:
                try:
                    data_0 = next(trainIter_0)
                except StopIteration:
                    trainIter_0  = iter(self.train_loader_1)
                    data_0 = next(trainIter_0)
                try:
                    data_1 = next(trainIter_1)
                except StopIteration:
                    trainIter_1  = iter(self.train_loader_2)
                    data_1 = next(trainIter_1)
                inputs_0      = self.convert_tensor_type(data_0['image'])
                labels_prob_0 = self.convert_tensor_type(data_0['label_prob'])       
                inputs_1      = self.convert_tensor_type(data_1['image'])
                inputs_11      = self.convert_tensor_type(data_1['image1']).to(self.device)
                labels_prob_1 = self.convert_tensor_type(data_1['label_prob'])   
                # sample_weight_1 = self.convert_tensor_type(data_1['image_weight'] ).to(self.device)
                # pixel_weight_1 = self.convert_tensor_type(data_1['pixel_weight'] ).to(self.device)
                inputs_0, labels_prob_0 = inputs_0.to(self.device), labels_prob_0.to(self.device)
                inputs_1, labels_prob_1 = inputs_1.to(self.device), labels_prob_1.to(self.device)
            elif number_domians == 1:
                try:
                    data_0 = next(trainIter_0)
                except StopIteration:
                    trainIter_0  = iter(self.train_loader_1)
                    data_0 = next(trainIter_0)
                inputs_0 = self.convert_tensor_type(data_0['image'])
                labels_prob_0 = self.convert_tensor_type(data_0['label_prob'])   
                inputs_0, labels_prob_0 = inputs_0.to(self.device), labels_prob_0.to(self.device)
        
            for train_idx in range(int(number_domians)):
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                if train_idx == 0:
                    
                    outputs_11 = self.net(inputs_0, domain_label=train_idx*torch.ones(inputs_0.shape[0], dtype=torch.long))
                    outputs_2222 = self.net(inputs_11, domain_label=train_idx*torch.ones(inputs_11.shape[0], dtype=torch.long))
                    outputs = outputs_11
                    loss = self.get_loss_value(data_0, outputs_11, labels_prob_0,self.fpl_uda)
                    loss2222 = self.get_loss_value(data_1, outputs_2222, labels_prob_1,self.fpl_uda)
                    loss += loss2222
                elif train_idx == 1:
                    outputs_22 = self.net(inputs_1, domain_label=train_idx*torch.ones(inputs_1.shape[0], dtype=torch.long))
                    outputs = outputs_22
                    loss = self.get_loss_value(data_1, outputs_22, labels_prob_1,self.fpl_uda)
                    if it>1000:
                        with torch.no_grad():
                            outputs_2222 = self.net(inputs_11, domain_label=0*torch.ones(inputs_11.shape[0], dtype=torch.long))
                        consis_loss = self.criterionL2(outputs_2222,outputs_22)
                        loss +=consis_loss
                   
                D,B,C,W,H = outputs.shape
                entropy1 = -(outputs.softmax(1) * torch.log2(outputs.softmax(1) + 1e-10)).sum()/(W*H*C*D)    
                # print(train_idx,entropy1.item(),'entropy1')
                loss += entropy1
                if(self.scheduler is not None and \
                    not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)):
                    self.scheduler.step()
                train_loss = train_loss + loss.item()
                # print(entropy1.item())
                # get dice evaluation for each class
                if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                    outputs = outputs[0] 
                outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                soft_out       = get_soft_label(outputs_argmax, class_num, self.tensor_type)
                if train_idx == 0:
                    soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob_0) 
                    dice_list = get_classwise_dice(soft_out, labels_prob)
                    train_dice_list_0.append(dice_list.cpu().numpy())

                elif train_idx == 1:
                    soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob_1) 
                    dice_list = get_classwise_dice(soft_out, labels_prob)
                    train_dice_list_1.append(dice_list.cpu().numpy())  
            if self.config['training']['dis']  == True:
                # self.optimizer.zero_grad()
                for train_idx in range(int(number_domians)):
                    # zero the parameter gradients
                    self.disSeg_opt.zero_grad()
                    # forward + backward + optimize
                    if train_idx == 0:
                        outputs_11,_ = self.net(inputs_0, domain_label=train_idx*torch.ones(inputs_0.shape[0], dtype=torch.long))
                        pred_real = self.disseg(outputs_11.softmax(1))
                        real = self.disseg(labels_prob_0)
                        # print(real.shape,pred_real.shape,labels_prob_0.shape,outputs_11.softmax(1).shape)
                        all1 = torch.ones_like(pred_real)
                        loss_real = self.criterionL2(pred_real, all1)
                        loss_lab = self.criterionL2(real, all1)
                        loss_dis = (loss_real+loss_lab)/2.0
                        loss_dis.backward()
                        self.disSeg_opt.step()
                        
                    elif train_idx == 1:
                        self.disSeg_opt.zero_grad()
                        outputs_22 = self.net(inputs_1, domain_label=train_idx*torch.ones(inputs_1.shape[0], dtype=torch.long))
                        pred_fake = self.disseg(outputs_22.softmax(1))      
                        all0 = torch.zeros_like(pred_fake)
                        loss_fake = self.criterionL2(pred_fake, all0)
                        # loss_D = (loss_real + loss_fake) * 0.5
                        loss_fake.backward()
                        self.disSeg_opt.step()
                # print('dis loss', (loss_fake.item(),loss_real.item()))

        train_avg_loss = train_loss / iter_valid / int(number_domians)
        train_cls_dice_0 = np.asarray(train_dice_list_0).mean(axis = 0)
        train_avg_dice_0 = train_cls_dice_0.mean()
        if number_domians == 2:
            train_cls_dice_1 = np.asarray(train_dice_list_1).mean(axis = 0)
            train_avg_dice_1 = train_cls_dice_1.mean()
            train_avg_dice = (train_avg_dice_0+train_avg_dice_1)/2
            train_cls_dice = (train_cls_dice_0+train_cls_dice_1)/2
        elif number_domians == 1:
            train_avg_dice = train_avg_dice_0
            train_cls_dice = train_cls_dice_0
        train_scalers = {'loss': train_avg_loss, 'avg_dice':train_avg_dice,'class_dice': train_cls_dice }
        return train_scalers
    def training(self):
        import tqdm
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        number_domians = self.config['network']['num_domains']
        train_loss  = 0
        train_dice_list_0 = []
        train_dice_list_1 = []
        self.net.train()
        self.criterionL2 = nn.MSELoss()
        if number_domians == 2:
            trainIter_0  = iter(self.train_loader_1)
            trainIter_1  = iter(self.train_loader_2)
        elif number_domians == 1:
            trainIter_0  = iter(self.train_loader_1)
        for it in range(iter_valid):
            if number_domians == 2:
                try:
                    data_0 = next(trainIter_0)
                except StopIteration:
                    trainIter_0  = iter(self.train_loader_1)
                    data_0 = next(trainIter_0)
                try:
                    data_1 = next(trainIter_1)
                except StopIteration:
                    trainIter_1  = iter(self.train_loader_2)
                    data_1 = next(trainIter_1)
                inputs_0      = self.convert_tensor_type(data_0['image'])
                labels_prob_0 = self.convert_tensor_type(data_0['label_prob'])       
                inputs_1      = self.convert_tensor_type(data_1['image'])
                labels_prob_1 = self.convert_tensor_type(data_1['label_prob'])   
                # sample_weight_1 = self.convert_tensor_type(data_1['image_weight'] ).to(self.device)
                # pixel_weight_1 = self.convert_tensor_type(data_1['pixel_weight'] ).to(self.device)
                inputs_0, labels_prob_0 = inputs_0.to(self.device), labels_prob_0.to(self.device)
                inputs_1, labels_prob_1 = inputs_1.to(self.device), labels_prob_1.to(self.device)
            elif number_domians == 1:
                try:
                    data_0 = next(trainIter_0)
                except StopIteration:
                    trainIter_0  = iter(self.train_loader_1)
                    data_0 = next(trainIter_0)
                inputs_0 = self.convert_tensor_type(data_0['image'])
                labels_prob_0 = self.convert_tensor_type(data_0['label_prob'])   
                inputs_0, labels_prob_0 = inputs_0.to(self.device), labels_prob_0.to(self.device)
        
            for train_idx in range(int(number_domians)):
                # zero the parameter gradients
                
                
                # forward + backward + optimize
                if train_idx == 0:
                    self.optimizer.zero_grad()
                    outputs_11 = self.net(inputs_0, domain_label=train_idx*torch.ones(inputs_0.shape[0], dtype=torch.long))
                    outputs = outputs_11
                    loss = self.get_loss_value(data_0, outputs_11, labels_prob_0,self.fpl_uda)
                elif train_idx == 1:
                    self.optimizer.zero_grad()
                    outputs_22 = self.net(inputs_1, domain_label=train_idx*torch.ones(inputs_0.shape[0], dtype=torch.long))
                    outputs = outputs_22
                    loss = self.get_loss_value(data_1, outputs_22, labels_prob_1,self.fpl_uda)
                   
                D,B,C,W,H = outputs.shape
                entropy1 = -(outputs.softmax(1) * torch.log2(outputs.softmax(1) + 1e-10)).sum()/(W*H*C*D)    
                loss += entropy1
                if(self.scheduler is not None and \
                    not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)):
                    self.scheduler.step()
                train_loss = train_loss + loss.item()
                # get dice evaluation for each class
                if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                    outputs = outputs[0] 
                outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                soft_out       = get_soft_label(outputs_argmax, class_num, self.tensor_type)
                if train_idx == 0:
                    soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob_0) 
                    dice_list = get_classwise_dice(soft_out, labels_prob)
                    train_dice_list_0.append(dice_list.cpu().numpy())

                elif train_idx == 1:
                    soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob_1) 
                    dice_list = get_classwise_dice(soft_out, labels_prob)
                    train_dice_list_1.append(dice_list.cpu().numpy())  
            if self.config['training']['dis']  == True:
                # self.optimizer.zero_grad()
                for train_idx in range(int(number_domians)):
                    # zero the parameter gradients
                    self.disSeg_opt.zero_grad()
                    # forward + backward + optimize
                    if train_idx == 0:
                        outputs_11,_ = self.net(inputs_0, domain_label=train_idx*torch.ones(inputs_0.shape[0], dtype=torch.long))
                        pred_real = self.disseg(outputs_11.softmax(1))
                        real = self.disseg(labels_prob_0)
                        # print(real.shape,pred_real.shape,labels_prob_0.shape,outputs_11.softmax(1).shape)
                        all1 = torch.ones_like(pred_real)
                        loss_real = self.criterionL2(pred_real, all1)
                        loss_lab = self.criterionL2(real, all1)
                        loss_dis = (loss_real+loss_lab)/2.0
                        loss_dis.backward()
                        self.disSeg_opt.step()
                        
                    elif train_idx == 1:
                        self.disSeg_opt.zero_grad()
                        outputs_22 = self.net(inputs_1, domain_label=train_idx*torch.ones(inputs_1.shape[0], dtype=torch.long))
                        pred_fake = self.disseg(outputs_22.softmax(1))      
                        all0 = torch.zeros_like(pred_fake)
                        loss_fake = self.criterionL2(pred_fake, all0)
                        # loss_D = (loss_real + loss_fake) * 0.5
                        loss_fake.backward()
                        self.disSeg_opt.step()
                # print('dis loss', (loss_fake.item(),loss_real.item()))

        train_avg_loss = train_loss / iter_valid / int(number_domians)
        train_cls_dice_0 = np.asarray(train_dice_list_0).mean(axis = 0)
        train_avg_dice_0 = train_cls_dice_0.mean()
        if number_domians == 2:
            train_cls_dice_1 = np.asarray(train_dice_list_1).mean(axis = 0)
            train_avg_dice_1 = train_cls_dice_1.mean()
            train_avg_dice = (train_avg_dice_0+train_avg_dice_1)/2
            train_cls_dice = (train_cls_dice_0+train_cls_dice_1)/2
        elif number_domians == 1:
            train_avg_dice = train_avg_dice_0
            train_cls_dice = train_cls_dice_0
        train_scalers = {'loss': train_avg_loss, 'avg_dice':train_avg_dice,'class_dice': train_cls_dice }
        return train_scalers
    def training_all(self):
        import tqdm
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        number_domians = self.config['network']['num_domains']
        train_loss  = 0
        train_dice_list_0 = []
        train_dice_list_1 = []
        self.net.train()
        self.criterionL2 = nn.MSELoss()
        if number_domians == 2:
            trainIter_0  = iter(self.train_loader_1)
            trainIter_1  = iter(self.train_loader_2)
        elif number_domians == 1:
            trainIter_0  = iter(self.train_loader_1)
        for it in range(iter_valid):
            if number_domians == 2:
                try:
                    data_0 = next(trainIter_0)
                except StopIteration:
                    trainIter_0  = iter(self.train_loader_1)
                    data_0 = next(trainIter_0)
                try:
                    data_1 = next(trainIter_1)
                except StopIteration:
                    trainIter_1  = iter(self.train_loader_2)
                    data_1 = next(trainIter_1)
                inputs_0      = self.convert_tensor_type(data_0['image'])
                labels_prob_0 = self.convert_tensor_type(data_0['label_prob'])       
                inputs_1      = self.convert_tensor_type(data_1['image'])
                labels_prob_1 = self.convert_tensor_type(data_1['label_prob'])   
                # sample_weight_1 = self.convert_tensor_type(data_1['image_weight'] ).to(self.device)
                # pixel_weight_1 = self.convert_tensor_type(data_1['pixel_weight'] ).to(self.device)
                inputs_0, labels_prob_0 = inputs_0.to(self.device), labels_prob_0.to(self.device)
                inputs_1, labels_prob_1 = inputs_1.to(self.device), labels_prob_1.to(self.device)
            elif number_domians == 1:
                try:
                    data_0 = next(trainIter_0)
                except StopIteration:
                    trainIter_0  = iter(self.train_loader_1)
                    data_0 = next(trainIter_0)
                inputs_0 = self.convert_tensor_type(data_0['image'])
                labels_prob_0 = self.convert_tensor_type(data_0['label_prob'])   
                inputs_0, labels_prob_0 = inputs_0.to(self.device), labels_prob_0.to(self.device)
            self.optimizer.zero_grad()
            for train_idx in range(int(number_domians)):
                # zero the parameter gradients
                # forward + backward + optimize
                
                if train_idx == 0:
                    outputs_11 = self.net(inputs_0, domain_label=train_idx*torch.ones(inputs_0.shape[0], dtype=torch.long))
                    D,B,C,W,H = outputs_11.shape
                    entropy1 = -(outputs_11.softmax(1) * torch.log2(outputs_11.softmax(1) + 1e-10)).sum()/(W*H*C*D) 
                    loss_11 = self.get_loss_value(data_0, outputs_11, labels_prob_0,self.fpl_uda)
                    loss = loss_11#+entropy1
                    if(isinstance(outputs_11, tuple) or isinstance(outputs_11, list)):
                        outputs_11 = outputs_11[0] 
                    outputs_argmax = torch.argmax(outputs_11, dim = 1, keepdim = True)
                    soft_out       = get_soft_label(outputs_argmax, class_num, self.tensor_type)
                    soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob_0) 
                    dice_list = get_classwise_dice(soft_out, labels_prob)
                    train_dice_list_0.append(dice_list.cpu().numpy())
                elif train_idx == 1:
                    outputs_22 = self.net(inputs_1, domain_label=train_idx*torch.ones(inputs_0.shape[0], dtype=torch.long))
                    entropy2 = -(outputs_22.softmax(1) * torch.log2(outputs_22.softmax(1) + 1e-10)).sum()/(W*H*C*D) 
                    loss_22 = self.get_loss_value(data_1, outputs_22, labels_prob_1,self.fpl_uda)
                    # loss = (loss + loss_22 + entropy2)/2
                    loss = (loss + loss_22)/2
                    if(isinstance(outputs_22, tuple) or isinstance(outputs_22, list)):
                        outputs_22 = outputs_22[0] 
                    outputs_argmax = torch.argmax(outputs_22, dim = 1, keepdim = True)
                    soft_out       = get_soft_label(outputs_argmax, class_num, self.tensor_type)
                    soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob_1) 
                    dice_list = get_classwise_dice(soft_out, labels_prob)
                    train_dice_list_1.append(dice_list.cpu().numpy())
            loss.backward()
            self.optimizer.step()
            if(self.scheduler is not None and \
                not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)):
                self.scheduler.step()
            train_loss = train_loss + loss.item()
        train_avg_loss = train_loss / iter_valid / int(number_domians)
        train_cls_dice_0 = np.asarray(train_dice_list_0).mean(axis = 0)
        train_avg_dice_0 = train_cls_dice_0.mean()
        if number_domians == 2:
            train_cls_dice_1 = np.asarray(train_dice_list_1).mean(axis = 0)
            train_avg_dice_1 = train_cls_dice_1.mean()
            train_avg_dice = (train_avg_dice_0+train_avg_dice_1)/2
            train_cls_dice = (train_cls_dice_0+train_cls_dice_1)/2
        elif number_domians == 1:
            train_avg_dice = train_avg_dice_0
            train_cls_dice = train_cls_dice_0
        train_scalers = {'loss': train_avg_loss, 'avg_dice':train_avg_dice,'class_dice': train_cls_dice }
        return train_scalers
    def validation(self):
        number_domians = self.config['network']['num_domains']
        class_num = self.config['network']['class_num']
        if(self.inferer is None):
            infer_cfg = self.config['testing']
            infer_cfg['class_num'] = class_num
            self.inferer = Inferer(infer_cfg)
        
        valid_loss_list_0 = []
        valid_dice_list_0 = []
        valid_loss_list_1 = []
        valid_dice_list_1 = []
        if number_domians == 2:
            validIter_0  = iter(self.valid_loader_1)
            validIter_1  = iter(self.valid_loader_2)
        elif number_domians == 1:
            validIter_0  = iter(self.valid_loader_1)
        with torch.no_grad():
            self.net.eval()
            for data in validIter_0:
                inputs      = self.convert_tensor_type(data['image'])
                labels_prob = self.convert_tensor_type(data['label_prob'])
                inputs, labels_prob  = inputs.to(self.device), labels_prob.to(self.device)
                
                batch_n = inputs.shape[0]
                outputs = self.inferer.run(self.net, inputs, domain_label=0*torch.ones(inputs.shape[0], dtype=torch.long))
                
                # The tensors are on CPU when calculating loss for validation data

                loss = self.get_loss_value(data, outputs, labels_prob)
                # print(self.get_loss_value(),'248')
                valid_loss_list_0.append(loss.item())

                if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                    outputs = outputs[0] 
                outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                soft_out  = get_soft_label(outputs_argmax, class_num, self.tensor_type)
                for i in range(batch_n):
                    soft_out_i, labels_prob_i = reshape_prediction_and_ground_truth(\
                        soft_out[i:i+1], labels_prob[i:i+1])
                    temp_dice = get_classwise_dice(soft_out_i, labels_prob_i)
                    valid_dice_list_0.append(temp_dice.cpu().numpy())
            if number_domians == 2:
                for data in validIter_1:
                    inputs      = self.convert_tensor_type(data['image'])
                    labels_prob = self.convert_tensor_type(data['label_prob'])
                    inputs, labels_prob  = inputs.to(self.device), labels_prob.to(self.device)
                    batch_n = inputs.shape[0]
                    outputs = self.inferer.run(self.net, inputs, domain_label=1*torch.ones(inputs.shape[0], dtype=torch.long))
                    
                    # The tensors are on CPU when calculating loss for validation data
                    loss = self.get_loss_value(data, outputs, labels_prob)
                    valid_loss_list_1.append(loss.item())

                    if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                        outputs = outputs[0] 
                    outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                    soft_out  = get_soft_label(outputs_argmax, class_num, self.tensor_type)
                    for i in range(batch_n):
                        soft_out_i, labels_prob_i = reshape_prediction_and_ground_truth(\
                            soft_out[i:i+1], labels_prob[i:i+1])
                        temp_dice = get_classwise_dice(soft_out_i, labels_prob_i)
                        valid_dice_list_1.append(temp_dice.cpu().numpy())
                

        valid_avg_loss_0 = np.asarray(valid_loss_list_0).mean()
        valid_cls_dice_0 = np.asarray(valid_dice_list_0).mean(axis = 0)
        valid_avg_dice_0 = valid_cls_dice_0.mean()
        # print(valid_loss_list_0,valid_dice_list_0,number_domians,'647')
        if number_domians == 1:
            valid_cls_dice = valid_cls_dice_0
            valid_avg_dice = valid_avg_dice_0
            valid_avg_loss = valid_avg_loss_0
            
        if number_domians == 2:
            valid_avg_loss_1 = np.asarray(valid_loss_list_1).mean()
            valid_cls_dice_1 = np.asarray(valid_dice_list_1).mean(axis = 0)
            # print('dice of domian 2:',valid_cls_dice_1)
            valid_avg_dice_1 = valid_cls_dice_1.mean()
            
            valid_cls_dice = (valid_cls_dice_0+valid_cls_dice_1) / 2
            valid_avg_dice = (valid_avg_dice_0+valid_avg_dice_1) / 2
            valid_avg_loss = (valid_avg_loss_0+valid_avg_loss_1) / 2
        if(isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)):
            self.scheduler.step(valid_avg_dice)
        if self.config['training']['val_t2']  == True:    
            valid_scalers = {'loss':valid_avg_loss_1, 'avg_dice': valid_avg_dice_1, \
            'class_dice': valid_cls_dice_1}
        elif self.config['training']['val_t1']  == True:    
            valid_scalers = {'loss':valid_avg_loss_0, 'avg_dice': valid_avg_dice_0, \
            'class_dice': valid_cls_dice_0}
        else:
            valid_scalers = {'loss':valid_avg_loss, 'avg_dice': valid_avg_dice, \
                'class_dice': valid_cls_dice}
            # print(valid_scalers,'664')
        return valid_scalers
  
    def validation_dual(self):
        number_domians = self.config['network']['num_domains']
        class_num = self.config['network']['class_num']
        if(self.inferer is None):
            infer_cfg = self.config['testing']
            infer_cfg['class_num'] = class_num
            self.inferer = Inferer(infer_cfg)
        
        valid_loss_list_0 = []
        valid_dice_list_0 = []
        
        if number_domians == 2:
            validIter_0  = iter(self.valid_loader_1)
        with torch.no_grad():
            # self.net.eval()
            for data in validIter_0:
                inputs      = self.convert_tensor_type(data['image'])
                labels_prob = self.convert_tensor_type(data['label_prob'])
                if self.train_fpl_uda:
                    image_weight = self.convert_tensor_type(data['image_weight'])
                    pixel_weight = self.convert_tensor_type(data['pixel_weight'])
                    image_weight, pixel_weight = image_weight.to(self.device), pixel_weight.to(self.device)
                inputs, labels_prob  = inputs.to(self.device), labels_prob.to(self.device)
                
                batch_n = inputs.shape[0]
                outputs = self.inferer.run(self.net, inputs, domain_label=1*torch.ones(inputs.shape[0], dtype=torch.long))
                # The tensors are on CPU when calculating loss for validation data

                loss = self.get_loss_value(data, outputs, labels_prob)
                # print(self.get_loss_value(),'248')
                valid_loss_list_0.append(loss.item())

                if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                    outputs = outputs[0] 
                outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                soft_out  = get_soft_label(outputs_argmax, class_num, self.tensor_type)

                
                for i in range(batch_n):
                    soft_out_i, labels_prob_i = reshape_prediction_and_ground_truth(\
                        soft_out[i:i+1], labels_prob[i:i+1])
                    temp_dice = get_classwise_dice(soft_out_i, labels_prob_i)
                    valid_dice_list_0.append(temp_dice.cpu().numpy())

                

        valid_avg_loss_0 = np.asarray(valid_loss_list_0).mean()
        valid_cls_dice_0 = np.asarray(valid_dice_list_0).mean(axis = 0)
        valid_avg_dice_0 = valid_cls_dice_0.mean()
        
        valid_cls_dice = valid_cls_dice_0
        valid_avg_dice = valid_avg_dice_0
        valid_avg_loss = valid_avg_loss_0
  
            
        
        if(isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)):
            self.scheduler.step(valid_avg_dice)
       
       
        valid_scalers = {'loss':valid_avg_loss, 'avg_dice': valid_avg_dice, \
            'class_dice': valid_cls_dice}
        return valid_scalers

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 'valid':valid_scalars['loss']}
        dice_scalar ={'train':train_scalars['avg_dice'], 'valid':valid_scalars['avg_dice']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        class_num = self.config['network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {'train':train_scalars['class_dice'][c], \
                'valid':valid_scalars['class_dice'][c]}
            self.summ_writer.add_scalars('class_{0:}_dice'.format(c), cls_dice_scalar, glob_it)
       
        logging.info('train loss {0:.4f}, avg dice {1:.4f} '.format(
            train_scalars['loss'], train_scalars['avg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in train_scalars['class_dice']) + "]")        
        logging.info('valid loss {0:.4f}, avg dice {1:.4f} '.format(
            valid_scalars['loss'], valid_scalars['avg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in valid_scalars['class_dice']) + "]")        

    def train_valid(self):
        self.dual  = self.config['training']['dual']
        self.fpl_uda = self.config['training']['train_fpl_uda']
        device_ids = self.config['training']['gpus']
        if(len(device_ids) > 1):
            self.device = torch.device("cuda:0")
            self.net = nn.DataParallel(self.net, device_ids = device_ids)
        else:
            self.device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(self.device)
                

        ckpt_dir    = self.config['training']['ckpt_save_dir']
        ckpt_prefix = self.config['training'].get('ckpt_prefix', None)
        if(ckpt_prefix is None):
            ckpt_prefix = ckpt_dir.split('/')[-1]
        iter_start  = self.config['training']['iter_start']
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training'].get('iter_save', None)
        early_stop_it = self.config['training'].get('early_stop_patience', None)
        if(iter_save is None):
            iter_save_list = [iter_max]
        elif(isinstance(iter_save, (tuple, list))):
            iter_save_list = iter_save
        else:
            iter_save_list = range(0, iter_max + 1, iter_save)

        self.max_val_dice = 0.0
        self.max_val_it   = 0
        self.best_model_wts = None 
        self.checkpoint = None
        if(iter_start > 0):
            checkpoint_file = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefix, iter_start)
            self.checkpoint = torch.load(checkpoint_file, map_location = self.device)
            self.checkpoint['valid_pred'] = 0
            # assert(self.checkpoint['iteration'] == iter_start)
            if(len(device_ids) > 1):
                self.net.module.load_state_dict(self.checkpoint['model_state_dict'])
            else:
                self.net.load_state_dict(self.checkpoint['model_state_dict'])
            self.max_val_dice = self.checkpoint.get('valid_pred', 0)

            # self.max_val_it   = self.checkpoint['iteration']
            self.max_val_it   = iter_start
            self.best_model_wts = self.checkpoint['model_state_dict']
            
        self.create_optimizer(self.get_parameters_to_update())
        self.create_loss_calculator()
    
        # self.trainIter  = iter(self.dataloader_train)
        
        logging.info("{0:} training start".format(str(datetime.now())[:-7]))
        self.summ_writer = SummaryWriter(self.config['training']['ckpt_save_dir'])
        self.glob_it = iter_start
        for it in range(iter_start, iter_max, iter_valid):
            lr_value = self.optimizer.param_groups[0]['lr']
            t0 = time.time()
            print('self.dual:',self.dual)
            if self.dual:
                train_scalars = self.training_all()
                t1 = time.time()
                valid_scalars = self.validation()
            else:
                train_scalars = self.training()
                # train_scalars = self.training_all()
                t1 = time.time()
                valid_scalars = self.validation()
            
            
            
            t2 = time.time()
            self.glob_it = it + iter_valid
            logging.info("\n{0:} it {1:}".format(str(datetime.now())[:-7], self.glob_it))
            logging.info('learning rate {0:}'.format(lr_value))
            logging.info("training/validation time: {0:.2f}s/{1:.2f}s".format(t1-t0, t2-t1))
            self.write_scalars(train_scalars, valid_scalars, lr_value, self.glob_it)
            # print(self.max_val_dice,'self.max_val_dice')
            if(valid_scalars['avg_dice'] > self.max_val_dice):
                
                self.max_val_dice = valid_scalars['avg_dice']
                self.max_val_it   = self.glob_it
                if(len(device_ids) > 1):
                    self.best_model_wts = copy.deepcopy(self.net.module.state_dict())
                else:
                    self.best_model_wts = copy.deepcopy(self.net.state_dict())

            stop_now = True if(early_stop_it is not None and \
                self.glob_it - self.max_val_it > early_stop_it) else False
            if self.config['training']['dis']  == True:
                
                if ((self.glob_it in iter_save_list) or stop_now):
                    save_dict = {'iteration': self.glob_it,
                                'valid_pred': valid_scalars['avg_dice'],
                                'model_state_dict': self.net.module.state_dict() \
                                    if len(device_ids) > 1 else self.net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'disseg_state_dict':self.disseg.state_dict()}
                    save_name = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefix, self.glob_it)
                    torch.save(save_dict, save_name) 
                    txt_file = open("{0:}/{1:}_latest.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                    txt_file.write(str(self.glob_it))
                    txt_file.close()
            else:
                if ((self.glob_it in iter_save_list) or stop_now):
                    save_dict = {'iteration': self.glob_it,
                                'valid_pred': valid_scalars['avg_dice'],
                                'model_state_dict': self.net.module.state_dict() \
                                    if len(device_ids) > 1 else self.net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()}
                    save_name = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefix, self.glob_it)
                    torch.save(save_dict, save_name) 
                    txt_file = open("{0:}/{1:}_latest.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                    txt_file.write(str(self.glob_it))
                    txt_file.close()
            if(stop_now):
                logging.info("The training is early stopped")
                break
        
        # save the best performing checkpoint
        if self.config['training']['dis']  == True:
            save_dict = {'iteration': self.max_val_it,
                        'valid_pred': self.max_val_dice,
                        'model_state_dict': self.best_model_wts,
                        'optimizer_state_dict': self.optimizer.state_dict(), 
                        'disseg_state_dict':self.disseg.state_dict()}
        else:
            save_dict = {'iteration': self.max_val_it,
                        'valid_pred': self.max_val_dice,
                        'model_state_dict': self.best_model_wts,
                        'optimizer_state_dict': self.optimizer.state_dict()}
        if self.config['training']['dis']  == True:
            torch.save(self.disseg.state_dict(),
                        "{0:}/{1:}_{2:}.pth".format(ckpt_dir, 'dis_para', self.max_val_it) )   
           
        save_name = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefix, self.max_val_it)
        torch.save(save_dict, save_name) 
        txt_file = open("{0:}/{1:}_best.txt".format(ckpt_dir, ckpt_prefix), 'wt')
        txt_file.write(str(self.max_val_it))
        txt_file.close()
        logging.info('The best performing iter is {0:}, valid dice {1:}'.format(\
            self.max_val_it, self.max_val_dice))
        self.summ_writer.close()
    

    def infer(self):
        domian_label = self.config['testing']['domian_label']
        device_ids = self.config['testing']['gpus']
        self.FPL = self.config['testing']['fpl']
        self.AE = self.config['testing']['ae']
        device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(device)
        
        # self.net.eva()
        if(self.config['testing'].get('evaluation_mode', True)):
            self.net.eval()
            if(self.config['testing'].get('test_time_dropout', False)) or self.FPL:

                def test_time_dropout(m):
                    if(type(m) == nn.Dropout):
                        logging.info('dropout layer')
                        m.train()
                print('FPL:',self.FPL,'***** FPL: dropout Turn on *****')
                self.net.apply(test_time_dropout)
                
        
        ckpt_mode = self.config['testing']['ckpt_mode']
        ckpt_name = self.get_checkpoint_name()
        if(ckpt_mode == 3):
            assert(isinstance(ckpt_name, (tuple, list)))
            self.infer_with_multiple_checkpoints()
            return 
        else:
            if(isinstance(ckpt_name, (tuple, list))):
                raise ValueError("ckpt_mode should be 3 if ckpt_name is a list")

        # load network parameters and set the network as evaluation mode
        checkpoint = torch.load(ckpt_name, map_location = device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        
        if(self.inferer is None):
            infer_cfg = self.config['testing']
            infer_cfg['class_num'] = self.config['network']['class_num']
            self.inferer = Inferer(infer_cfg)
        postpro_name = self.config['testing'].get('post_process', None)
        if(self.postprocessor is None and postpro_name is not None):
            self.postprocessor = PostProcessDict[postpro_name](self.config['testing'])
        infer_time_list = []
        boundary = 0
        uncertainty_list = {}
        pixel_weight = {}
        with torch.no_grad():
            for data in self.test_loader:
                images = self.convert_tensor_type(data['image'])
                images = images.to(device)
    
                # for debug
                # for i in range(images.shape[0]):
                #     image_i = images[i][0]
                #     label_i = images[i][0]
                #     image_name = "temp/{0:}_image.nii.gz".format(names[0])
                #     label_name = "temp/{0:}_label.nii.gz".format(names[0])
                #     save_nd_array_as_image(image_i, image_name, reference_name = None)
                #     save_nd_array_as_image(label_i, label_name, reference_name = None)
                # continue
                start_time = time.time()
                maps = []
                hards = []
                if self.FPL:
                    for i in range(6):
                        pred = self.inferer.run(self.net, images, domain_label=domian_label*torch.ones(images.shape[0], dtype=torch.long))
                        if(isinstance(pred, (tuple, list))):
                            pred = [item.cpu().numpy() for item in pred]
                        else:
                            pred = pred.cpu().numpy()
                        data['predict'] = pred
                        for transform in self.transform_list[::-1]:
                            if (transform.inverse):
                                data = transform.inverse_transform_for_prediction(data) 
                        names, pred = data['names'], data['predict']
                        if(isinstance(pred, (list, tuple))):
                            pred =  pred[0]
                        prob   = scipy.special.softmax(pred, axis = 1) 
                        # uncertainty = -1.0 * (prob[:,1]*np.log(prob[:,1] + 1e-6))
                        # boundary += np.where(uncertainty > 0.2 ,1,0).sum()
                        hard = np.asarray(np.argmax(prob,  axis = 1), np.uint8)
                        if i ==0:
                            maps = prob
                            hards = hard
                        else: 
                            maps = np.concatenate((maps,prob),axis=0) 
                            hards = np.concatenate((hards,hard),axis=0) 
                    vars = maps.var(axis=0).sum()
                    means = np.mean(maps[:,1],axis=0)
                    uncertainty = -1.0 * (means*np.log(means + 1e-6))
                    boundary = np.where(uncertainty > 0.01 ,1,0).sum()
                    pixel_weight[names[0]] = [uncertainty]
                    if boundary < 50:
                        uncer_one = 1
                    else:
                        uncer_one = vars/(boundary)
                    print(names[0],uncer_one)
                    uncertainty_list[names[0]] = [uncer_one]
                        
                else:                    
                    pred = self.inferer.run(self.net, images, domain_label=domian_label*torch.ones(images.shape[0], dtype=torch.long))
                    

                    # convert tensor to numpy
                    if(isinstance(pred, (tuple, list))):
                        pred = [item.cpu().numpy() for item in pred]
                    else:
                        pred = pred.cpu().numpy()
                    
                    data['predict'] = pred
                    # inverse transform
                    for transform in self.transform_list[::-1]:
                        if (transform.inverse):
                            data = transform.inverse_transform_for_prediction(data) 
                        

                    infer_time = time.time() - start_time
                    infer_time_list.append(infer_time)
                    
                    self.save_outputs(data)
        if self.FPL:
            # uncertainty_list
            print('the max model is: ',max(uncertainty_list, key=uncertainty_list.get))
            student_tuplelist = list(zip(uncertainty_list.values(),uncertainty_list.keys()))
            student_tuplelist_sorted = sorted(student_tuplelist, reverse=False)
            print(student_tuplelist_sorted)
            np.save(self.config['testing']['fpl_uncertainty_sorted'], student_tuplelist_sorted)
            # np.save(self.config['testing']['fpl_uncertainty_weight'], pixel_weight)
        infer_time_list = np.asarray(infer_time_list)
        time_avg, time_std = infer_time_list.mean(), infer_time_list.std()
        logging.info("testing time {0:} +/- {1:}".format(time_avg, time_std))

    def infer_with_multiple_checkpoints(self):
        """
        Inference with ensemble of multilple check points.
        """
        device_ids = self.config['testing']['gpus']
        device = torch.device("cuda:{0:}".format(device_ids[0]))
        domian_label = self.config['testing']['domian_label']
        if(self.inferer is None):
            infer_cfg  = self.config['testing']
            infer_cfg['class_num'] = self.config['network']['class_num']
            self.inferer = Inferer(infer_cfg)
        ckpt_names = self.config['testing']['ckpt_name']
        infer_time_list = []
        with torch.no_grad():
            for data in self.test_loader:
                images = self.convert_tensor_type(data['image'])
                images = images.to(device)
    
                # for debug
                # for i in range(images.shape[0]):
                #     image_i = images[i][0]
                #     label_i = images[i][0]
                #     image_name = "temp/{0:}_image.nii.gz".format(names[0])
                #     label_name = "temp/{0:}_label.nii.gz".format(names[0])
                #     save_nd_array_as_image(image_i, image_name, reference_name = None)
                #     save_nd_array_as_image(label_i, label_name, reference_name = None)
                # continue
                start_time = time.time()
                predict_list = []

                for ckpt_name in ckpt_names:
                    checkpoint = torch.load(ckpt_name, map_location = device)
                    self.net.load_state_dict(checkpoint['model_state_dict'])
                    
                    pred = self.inferer.run(self.net, images, domain_label=domian_label*torch.ones(images.shape[0], dtype=torch.long))
                    # convert tensor to numpy
                    if(isinstance(pred, (tuple, list))):
                        pred = [item.cpu().numpy() for item in pred]
                    else:
                        pred = pred.cpu().numpy()
                    predict_list.append(pred)
                pred = np.mean(predict_list, axis=0)

                data['predict'] = pred
                # inverse transform
                for transform in self.transform_list[::-1]:
                    if (transform.inverse):
                        data = transform.inverse_transform_for_prediction(data) 
                
                infer_time = time.time() - start_time
                infer_time_list.append(infer_time)
                self.save_outputs(data)
        infer_time_list = np.asarray(infer_time_list)
        time_avg, time_std = infer_time_list.mean(), infer_time_list.std()
        logging.info("testing time {0:} +/- {1:}".format(time_avg, time_std))

    def save_outputs(self, data):
        """
        Save prediction output. 

        :param data: (dictionary) A data dictionary with prediciton result and other 
            information such as input image name. 
        """
        output_dir = self.config['testing']['output_dir']
        ignore_dir = self.config['testing'].get('filename_ignore_dir', True)
        save_prob  = self.config['testing'].get('save_probability', False)
        label_source = self.config['testing'].get('label_source', None)
        label_target = self.config['testing'].get('label_target', None)
        filename_replace_source = self.config['testing'].get('filename_replace_source', None)
        filename_replace_target = self.config['testing'].get('filename_replace_target', None)
        ckpt_dir    = self.config['training']['ckpt_save_dir']
        ckpt_dir = ckpt_dir.split('/')[-1]
        subset = self.config['dataset']['test_csv']
        subset = subset.split('/')[-1][:-4]
        output_dir = os.path.join(output_dir,ckpt_dir+'_'+subset)
        self.output_dir = output_dir
        if(not os.path.exists(output_dir)):
            os.makedirs(output_dir, exist_ok=True)

        names, pred = data['names'], data['predict']
        # print(pred.sum(),'1565')
        if(isinstance(pred, (list, tuple))):
            pred =  pred[0]
        prob   = scipy.special.softmax(pred, axis = 1) 
        output = np.asarray(np.argmax(prob,  axis = 1), np.uint8)
        # print(output.shape,output.sum())
        if((label_source is not None) and (label_target is not None)):
            output = convert_label(output, label_source, label_target)
        if(self.postprocessor is not None):
            for i in range(len(names)):
                output[i] = self.postprocessor(output[i])
        # save the output and (optionally) probability predictions
        root_dir  = self.config['dataset']['root_dir']
        for i in range(len(names)):
            save_name = names[i].split('/')[-1] if ignore_dir else \
                names[i].replace('/', '_')
            if((filename_replace_source is  not None) and (filename_replace_target is not None)):
                save_name = save_name.replace(filename_replace_source, filename_replace_target)
            save_name = "{0:}/{1:}".format(output_dir, save_name)
            save_nd_array_as_image(output[i], save_name, root_dir + '/' + names[i])
            save_name_split = save_name.split('.')

            # if(not save_prob):
            #     continue
            # if('.nii.gz' in save_name):
            #     save_prefix = '.'.join(save_name_split[:-2])
            #     save_format = 'nii.gz'
            # else:
            #     save_prefix = '.'.join(save_name_split[:-1])
            #     save_format = save_name_split[-1]
            
            # class_num = prob.shape[1]
            # for c in range(0, class_num):
            #     temp_prob = prob[i][c]
            #     prob_save_name = "{0:}_prob_{1:}.{2:}".format(save_prefix, c, save_format)
            #     if(len(temp_prob.shape) == 2):
            #         temp_prob = np.asarray(temp_prob * 255, np.uint8)
            #     save_nd_array_as_image(temp_prob, prob_save_name, root_dir + '/' + names[i])
