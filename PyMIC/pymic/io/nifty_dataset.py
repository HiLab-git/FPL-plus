# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pymic.io.image_read_write import load_image_as_nd_array



class NiftyDataset_dual(Dataset):
    """
    Dataset for loading images for segmentation. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors 
    with dimention order [C, H, W] for 2D images.

    :param root_dir: (str) Directory with all the images. 
    :param csv_file: (str) Path to the csv file with image names.
    :param modal_num: (int) Number of modalities. 
    :param with_label: (bool) Load the data with segmentation ground truth or not.
    :param transform:  (list) List of transforms to be applied on a sample.
        The built-in transforms can listed in :mod:`pymic.transform.trans_dict`.
    """
    def __init__(self, root_dir, csv_file, modal_num = 1, 
            with_label = False, transform=None):
        self.root_dir   = root_dir
        self.csv_items  = pd.read_csv(csv_file)
        self.modal_num  = modal_num
        self.with_label = with_label
        self.transform  = transform
       
        csv_keys = list(self.csv_items.keys())
        # print(csv_keys,'csv keys 36 ')
        self.image_weight_idx = None
        self.pixel_weight_idx = None
        self.image1 = None
        if('image_weight' in csv_keys):
            self.image_weight_idx = csv_keys.index('image_weight')
        if('pixel_weight' in csv_keys):
            self.pixel_weight_idx = csv_keys.index('pixel_weight')
        if('pixel_weight_nonl' in csv_keys):
            self.image1 = csv_keys.index('pixel_weight_nonl')

    def __len__(self):
        return len(self.csv_items)

    def __getlabel__(self, idx):
        csv_keys = list(self.csv_items.keys())
        label_idx = csv_keys.index('label')
        label_name = "{0:}/{1:}".format(self.root_dir, 
            self.csv_items.iloc[idx, label_idx])
        label = load_image_as_nd_array(label_name)['data_array']
        label = np.asarray(label, np.int32)
        # print(label.shape,'51')
        return label

    def __get_pixel_weight__(self, idx):
        weight_name = "{0:}/{1:}".format(self.root_dir, 
            self.csv_items.iloc[idx, self.pixel_weight_idx])
        weight = load_image_as_nd_array(weight_name)['data_array']
        weight = np.asarray(weight, np.float32)
        return weight        
    def __get_pixel_weight_nonl__(self, idx):
        weight_name = "{0:}/{1:}".format(self.root_dir, 
            self.csv_items.iloc[idx, self.image1])
        weight = load_image_as_nd_array(weight_name)['data_array']
        weight = np.asarray(weight, np.float32)
        return weight  

    def __getitem__(self, idx):
        names_list, image_list = [], []
        for i in range (self.modal_num):
            image_name = self.csv_items.iloc[idx, i]
            image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
            image_dict = load_image_as_nd_array(image_full_name)
            image_data = image_dict['data_array']
            names_list.append(image_name)
            image_list.append(image_data)
        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)    
        # print(image.max(),image.min(),'11111')
        sample = {'image': image, 'names' : names_list[0], 
                 'origin':image_dict['origin'],
                 'spacing': image_dict['spacing'],
                 'direction':image_dict['direction']}
        if (self.with_label):   
            sample['label'] = self.__getlabel__(idx) 
            assert(image.shape[1:] == sample['label'].shape[1:])
        if (self.image_weight_idx is not None):
            sample['image_weight'] = self.csv_items.iloc[idx, self.image_weight_idx]
        if (self.pixel_weight_idx is not None):
            sample['pixel_weight'] = self.__get_pixel_weight__(idx) 
            assert(image.shape[1:] == sample['pixel_weight'].shape[1:])
        if (self.image1 is not None):
            sample['pixel_weight'] = self.__get_pixel_weight_nonl__(idx) 
            assert(image.shape[1:] == sample['pixel_weight'].shape[1:])
            print(sample['pixel_weight'].shape,'pixel weight')
        if self.transform:
            sample = self.transform(sample)

        return sample

class NiftyDataset(Dataset):
    """
    Dataset for loading images for segmentation. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors 
    with dimention order [C, H, W] for 2D images.

    :param root_dir: (str) Directory with all the images. 
    :param csv_file: (str) Path to the csv file with image names.
    :param modal_num: (int) Number of modalities. 
    :param with_label: (bool) Load the data with segmentation ground truth or not.
    :param transform:  (list) List of transforms to be applied on a sample.
        The built-in transforms can listed in :mod:`pymic.transform.trans_dict`.
    """
    def __init__(self, root_dir, csv_file, modal_num = 1, 
            with_label = False, transform=None):
        self.root_dir   = root_dir
        print(csv_file,'28')
        self.csv_items  = pd.read_csv(csv_file)
        self.modal_num  = modal_num
        self.with_label = with_label
        self.transform  = transform
       
        csv_keys = list(self.csv_items.keys())
        # print(csv_keys,'csv keys 36 ')
        self.image_weight_idx = None
        self.pixel_weight_idx = None
        self.image1 = None
        if('image_weight' in csv_keys):
            self.image_weight_idx = csv_keys.index('image_weight')
        if('pixel_weight' in csv_keys):
            self.pixel_weight_idx = csv_keys.index('pixel_weight')
        if('image1' in csv_keys):
            self.image1 = csv_keys.index('image1')

    def __len__(self):
        return len(self.csv_items)

    def __getlabel__(self, idx):
        csv_keys = list(self.csv_items.keys())
        label_idx = csv_keys.index('label')
        label_name = "{0:}/{1:}".format(self.root_dir, 
            self.csv_items.iloc[idx, label_idx])
        label = load_image_as_nd_array(label_name)['data_array']
        label = np.asarray(label, np.int32)
        # print(label.shape,'51')
        return label

    def __get_pixel_weight__(self, idx):
        weight_name = "{0:}/{1:}".format(self.root_dir, 
            self.csv_items.iloc[idx, self.pixel_weight_idx])
        weight = load_image_as_nd_array(weight_name)['data_array']
        weight = np.asarray(weight, np.float32)
        return weight        
    def __get_image1__(self, idx):
        weight_name = "{0:}/{1:}".format(self.root_dir, 
            self.csv_items.iloc[idx, self.image1])
        weight = load_image_as_nd_array(weight_name)['data_array']
        weight = np.asarray(weight, np.float32)
        return weight  
    def set_weight_(self,img_weight,pixel_weight):
        pixel_weight[pixel_weight < 1] = 0
        pixel_weight = pixel_weight * img_weight   
        return pixel_weight


    def __getitem__(self, idx):
        names_list, image_list = [], []
        for i in range (self.modal_num):
            image_name = self.csv_items.iloc[idx, i]
            image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
            image_dict = load_image_as_nd_array(image_full_name)
            image_data = image_dict['data_array']
            names_list.append(image_name)
            image_list.append(image_data)
        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32) 
        
        sample = {'image': image, 'names' : names_list[0], 
                 'origin':image_dict['origin'],
                 'spacing': image_dict['spacing'],
                 'direction':image_dict['direction']}
        if (self.with_label):   
            sample['label'] = self.__getlabel__(idx) 
            # print(image.shape,sample['label'].shape,'201',image_name)
            assert(image.shape[1:] == sample['label'].shape[1:])
        if (self.image_weight_idx is not None):
            sample['image_weight'] = self.csv_items.iloc[idx, self.image_weight_idx]
            if (self.pixel_weight_idx is None):
                sample['pixel_weight'] = np.ones_like(image)
                sample['pixel_weight'] = self.set_weight_(sample['image_weight'],sample['pixel_weight'])
                # print('213')
        if (self.pixel_weight_idx is not None):
            try:
                sample['pixel_weight'] = self.__get_pixel_weight__(idx) 
                sample['pixel_weight'] = self.set_weight_(sample['image_weight'],sample['pixel_weight'])
            except:
                sample['pixel_weight'] = np.ones_like(image)*0.5
                print('219*****')
            assert(image.shape[1:] == sample['pixel_weight'].shape[1:])
            
        if (self.image1 is not None):
            try:
                sample['image1'] = self.__get_image1__(idx) 
                assert(image.shape[1:] == sample['pixel_weight'].shape[1:])
            except:
                sample['image1'] = image
            
            # print(sample['pixel_weight'].shape,'pixel weight')
        if self.transform:
            sample = self.transform(sample)
        # if (self.pixel_weight_idx is not None):
        #     sample['pixel_weight'] = self.set_weight_(sample['image_weight'],sample['pixel_weight'])
        return sample
        
class NiftyDataset_npy(Dataset):
    """
    Dataset for loading images for segmentation. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors 
    with dimention order [C, H, W] for 2D images.

    :param root_dir: (str) Directory with all the images. 
    :param csv_file: (str) Path to the csv file with image names.
    :param modal_num: (int) Number of modalities. 
    :param with_label: (bool) Load the data with segmentation ground truth or not.
    :param transform:  (list) List of transforms to be applied on a sample.
        The built-in transforms can listed in :mod:`pymic.transform.trans_dict`.
    """
    def __init__(self, root_dir, csv_file, modal_num = 1, train_fpl_uda=False,
            with_label = False, transform=None):
        self.root_dir   = root_dir
        # print(csv_file,'28')
        self.csv_items  = pd.read_csv(csv_file)
        self.modal_num  = modal_num
        self.with_label = with_label
        self.transform  = transform
        self.train_fpl_uda = train_fpl_uda
        csv_keys = list(self.csv_items.keys())
        # if train_fpl_uda:
        #     self.image_weight_idx = True
        #     self.pixel_weight_idx = True
        # else:
        self.image_weight_idx = None
        self.pixel_weight_idx = None
        # if('image_weight' in csv_keys):
        #     self.image_weight_idx = csv_keys.index('image_weight')
        # if('pixel_weight' in csv_keys):
        #     self.pixel_weight_idx = csv_keys.index('pixel_weight')

    def __len__(self):
        return len(self.csv_items)

    def __getlabel__(self, idx):
        csv_keys = list(self.csv_items.keys())
        label_idx = csv_keys.index('label')
        label_name = "{0:}/{1:}".format(self.root_dir, 
            self.csv_items.iloc[idx, label_idx])
        if self.train_fpl_uda:
            # print(label_name,'55')
            label = load_image_as_nd_array(label_name).item()
            # print(label.keys,'--------------------------------')
            label = np.asarray(label['predict'], np.int32)
            weight_pixel = load_image_as_nd_array(label_name).item()['pixel_wise_weight']
            weight_pixel = np.asarray(weight_pixel, np.float32)
            weight_sample = load_image_as_nd_array(label_name).item()['sample_wise_weight']
            weight_pixel = np.expand_dims(np.asarray(weight_pixel, np.float32),axis=0)
            # print(wei)
            return label, weight_pixel, weight_sample
        else:
            label = load_image_as_nd_array(label_name)['data_array']
            label = np.asarray(label, np.int32)
            # print(label.shape,'51')
            return label

    def __get_pixel_weight__(self, idx):
        weight_name = "{0:}/{1:}".format(self.root_dir, 
            self.csv_items.iloc[idx, self.pixel_weight_idx])
        if self.train_fpl_uda:
            weight_pixel = load_image_as_nd_array(weight_name)['pixel_wise_weight']
            weight_pixel = np.asarray(weight, np.float32)
            weight_sample = load_image_as_nd_array(weight_name)['sample_wise_weight']
            weight_sample = np.asarray(weight, np.float32)
            return weight_pixel,weight_sample
        else:
            weight = load_image_as_nd_array(weight_name)['data_array']
            weight = np.asarray(weight, np.float32)
            return weight        
    
     

    def __getitem__(self, idx):
        names_list, image_list = [], []
        for i in range (self.modal_num):
            image_name = self.csv_items.iloc[idx, i]
            image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
            image_dict = load_image_as_nd_array(image_full_name)
            image_data = image_dict['data_array']
            names_list.append(image_name)
            image_list.append(image_data)
        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)    
        sample = {'image': image, 'names' : names_list[0], 
                 'origin':image_dict['origin'],
                 'spacing': image_dict['spacing'],
                 'direction':image_dict['direction']}
        if (self.with_label):   
            sample['label'],sample['pixel_weight'], sample['image_weight'] = self.__getlabel__(idx) 
            # print(sample['label'].shape,sample['pixel_weight'].shape)
            assert(image.shape[1:] == sample['label'].shape[1:])
        # if (self.image_weight_idx is not None):
        #     self.csv_items.iloc[idx, self.image_weight_idx]
        if (self.pixel_weight_idx is not None):
            sample['pixel_weight'], sample['image_weight']  = self.__get_pixel_weight__(idx) 
            assert(image.shape[1:] == sample['pixel_weight'].shape[1:])
        if self.transform:
            
            sample = self.transform(sample)
            # print('111****************')
            # print(sample['image'].shape,sample['label'].shape,sample['pixel_weight'].shape)
        return sample


class ClassificationDataset(NiftyDataset):
    """
    Dataset for loading images for classification. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors 
    with dimention order [C, H, W] for 2D images.

    :param root_dir: (str) Directory with all the images. 
    :param csv_file: (str) Path to the csv file with image names.
    :param modal_num: (int) Number of modalities. 
    :param class_num: (int) Class number of the classificaiton task.
    :param with_label: (bool) Load the data with segmentation ground truth or not.
    :param transform:  (list) List of transforms to be applied on a sample.
        The built-in transforms can listed in :mod:`pymic.transform.trans_dict`.
    """
    def __init__(self, root_dir, csv_file, modal_num = 1, class_num = 2, 
            with_label = False, transform=None):
        super(ClassificationDataset, self).__init__(root_dir, 
            csv_file, modal_num, with_label, transform)
        self.class_num = class_num

    def __getlabel__(self, idx):
        csv_keys = list(self.csv_items.keys())
        label_idx = csv_keys.index('label')
        label = self.csv_items.iloc[idx, label_idx]
        return label
    
    def __getweight__(self, idx):
        weight = self.csv_items.iloc[idx, self.image_weight_idx]
        weight = weight + 0.0
        return weight   
    
    def __getitem__(self, idx):
        names_list, image_list = [], []
        for i in range (self.modal_num):
            image_name = self.csv_items.iloc[idx, i]
            image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
            image_dict = load_image_as_nd_array(image_full_name)
            image_data = image_dict['data_array']
            names_list.append(image_name)
            image_list.append(image_data)
        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)    
        sample = {'image': image, 'names' : names_list[0], 
                 'origin':image_dict['origin'],
                 'spacing': image_dict['spacing'],
                 'direction':image_dict['direction']}
        if (self.with_label):   
            sample['label'] = self.__getlabel__(idx) 
        if (self.image_weight_idx is not None):
            sample['image_weight'] = self.__getweight__(idx) 
        if self.transform:
            sample = self.transform(sample)
        return sample
