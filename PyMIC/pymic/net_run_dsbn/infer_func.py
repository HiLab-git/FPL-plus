# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
from torch.nn.functional import interpolate

class Inferer(object):
    """
    The class for inference.
    The arguments should be written in the `config` dictionary, 
    and it has the following fields:

    :param `sliding_window_enable`: (optional, bool) Default is `False`.
    :param `sliding_window_size`: (optional, list) The sliding window size. 
    :param `sliding_window_stride`: (optional, list) The sliding window stride. 
    :param `tta_mode`: (optional, int) The test time augmentation mode. Default
        is 0 (no test time augmentation). The other option is 1 (augmentation 
        with horinzontal and vertical flipping).
    """
    def __init__(self, config):
        self.config = config
        
    def __infer(self,image,domain_label):
        use_sw  = self.config.get('sliding_window_enable', False)
        if(not use_sw):
            outputs = self.model(image,domain_label=domain_label)
        else:
            outputs = self.__infer_with_sliding_window(image,domain_label)
        return outputs

    def __get_prediction_number_and_scales(self, tempx, domain_label):
        """
        If the network outputs multiple tensors with different sizes, return the
        number of tensors and the scale of each tensor compared with the first one
        """
        img_dim = len(tempx.shape) - 2
        output = self.model(tempx,domain_label=domain_label)
        if(isinstance(output, (tuple, list))):
            output_num = len(output)
            scales = [[1.0] * img_dim]
            shape0 = list(output[0].shape[2:])
            for  i in range(1, output_num):
                shapei= list(output[i].shape[2:])
                scale = [(shapei[d] + 0.0) / shape0[d] for d in range(img_dim)]
                scales.append(scale)
        else:
            output_num, scales = 1, None
        return output_num, scales

    def __infer_with_sliding_window(self, image,domain_label):
        """
        Use sliding window to predict segmentation for large images.
        Note that the network may output a list of tensors with difference sizes.
        """
        window_size   = [x for x in self.config['sliding_window_size']]
        window_stride = [x for x in self.config['sliding_window_stride']]
        class_num     = self.config['class_num']
        img_full_shape = list(image.shape)
        batch_size = img_full_shape[0]
        img_shape  = img_full_shape[2:]
        img_dim    = len(img_shape)
        if(img_dim != 2 and img_dim !=3):
            raise ValueError("Inference using sliding window only supports 2D and 3D images")

        for d in range(img_dim):
            if (window_size[d] is None) or window_size[d] > img_shape[d]:
                window_size[d]  = img_shape[d]
            if (window_stride[d] is None) or window_stride[d] > window_size[d]:
                window_stride[d] = window_size[d]
                
        if all([window_size[d] >= img_shape[d] for d in range(img_dim)]):
            output = self.model(image,domain_label)
            return output

        crop_start_list  = []
        for w in range(0, img_shape[-1], window_stride[-1]):
            w_min = min(w, img_shape[-1] - window_size[-1])
            for h in range(0, img_shape[-2], window_stride[-2]):
                h_min = min(h, img_shape[-2] - window_size[-2])
                if(img_dim == 2):
                    crop_start_list.append([h_min, w_min])
                else:
                    for d in range(0, img_shape[0], window_stride[0]):
                        d_min = min(d, img_shape[0] - window_size[0])
                        crop_start_list.append([d_min, h_min, w_min])

        output_shape = [batch_size, class_num] + img_shape
        mask_shape   = [batch_size, class_num] + window_size
        counter      = torch.zeros(output_shape).to(image.device)
        temp_mask    = torch.ones(mask_shape).to(image.device)
        temp_in_shape = img_full_shape[:2] + window_size
        tempx = torch.ones(temp_in_shape).to(image.device)
        out_num, scale_list = self.__get_prediction_number_and_scales(tempx,domain_label=domain_label)
        if(out_num == 1): # for a single prediction
            output = torch.zeros(output_shape).to(image.device)
            for c0 in crop_start_list:
                c1 = [c0[d] + window_size[d] for d in range(img_dim)]
                if(img_dim == 2):
                    patch_in = image[:, :, c0[0]:c1[0], c0[1]:c1[1]]
                else:
                    patch_in = image[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
                patch_out = self.model(patch_in,domain_label=domain_label) 

                if(isinstance(patch_out, (tuple, list))):
                    patch_out = patch_out[0]
                if(img_dim == 2):
                    output[:, :, c0[0]:c1[0], c0[1]:c1[1]] += patch_out
                    counter[:, :, c0[0]:c1[0], c0[1]:c1[1]] += temp_mask
                else:
                    output[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] += patch_out
                    counter[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] += temp_mask
            return output/counter
        else: # for multiple prediction
            output_list= []
            for i in range(out_num):
                output_shape_i = [batch_size, class_num] + \
                    [int(img_shape[d] * scale_list[i][d]) for d in range(img_dim)]
                output_list.append(torch.zeros(output_shape_i).to(image.device))

            for c0 in crop_start_list:
                c1 = [c0[d] + window_size[d] for d in range(img_dim)]
                if(img_dim == 2):
                    patch_in = image[:, :, c0[0]:c1[0], c0[1]:c1[1]]
                else:
                    patch_in = image[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
                patch_out = self.model(patch_in,domain_label=domain_label) 

                for i in range(out_num):
                    c0_i = [int(c0[d] * scale_list[i][d]) for d in range(img_dim)]
                    c1_i = [int(c1[d] * scale_list[i][d]) for d in range(img_dim)]
                    if(img_dim == 2):
                        output_list[i][:, :, c0_i[0]:c1_i[0], c0_i[1]:c1_i[1]] += patch_out[i]
                        counter[:, :, c0[0]:c1[0], c0[1]:c1[1]] += temp_mask
                    else:
                        output_list[i][:, :, c0_i[0]:c1_i[0], c0_i[1]:c1_i[1], c0_i[2]:c1_i[2]] += patch_out[i]
                        counter[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] += temp_mask
            for i in range(out_num):  
                counter_i = interpolate(counter, scale_factor = scale_list[i])
                output_list[i] = output_list[i] / counter_i
            return output_list

    def run_flip(self, model, image,i, domain_label):
        """
        Using `model` for inference on `image`.

        :param model: (nn.Module) a network.
        :param image: (tensor) An image.
        """
        self.model = model
        tta_mode   = self.config.get('tta_mode', 0)
        if(tta_mode == 0):
            outputs = self.__infer(image,domain_label)
        elif(tta_mode == 1): 
            # test time augmentation with flip in 2D
            # you may define your own method for test time augmentation
            outputs1 = self.__infer(image,domain_label=domain_label)
            outputs2 = self.__infer(torch.flip(image, [-2]),domain_label)
            outputs3 = self.__infer(torch.flip(image, [-1]),domain_label)
            outputs4 = self.__infer(torch.flip(image, [-2, -1]),domain_label)
            if(isinstance(outputs1, (tuple, list))):
                outputs = []
                for i in range(len(outputs1)):
                    temp_out1 = outputs1[i]
                    temp_out2 = torch.flip(outputs2[i], [-2])
                    temp_out3 = torch.flip(outputs3[i], [-1])
                    temp_out4 = torch.flip(outputs4[i], [-2, -1])
                    temp_mean = (temp_out1 + temp_out2 + temp_out3 + temp_out4) / 4
                    outputs.append(temp_mean)
            else:
                outputs2 = torch.flip(outputs2, [-2])
                outputs3 = torch.flip(outputs3, [-1])
                outputs4 = torch.flip(outputs4, [-2, -1])
                outputs = (outputs1 + outputs2 + outputs3 + outputs4) / 4
        else:
            raise ValueError("Undefined tta_mode {0:}".format(tta_mode))
        if i ==0:
            return outputs1
        elif i == 1:
            return outputs2
        elif i == 2:
            return outputs3
        elif i == 3:
            return outputs4
        elif i == 5:
            return outputs
        else:
            return outputs
    def run(self, model, image, domain_label):
        """
        Using `model` for inference on `image`.

        :param model: (nn.Module) a network.
        :param image: (tensor) An image.
        """
        self.model = model
        tta_mode   = self.config.get('tta_mode', 0)
        if(tta_mode == 0):
            outputs = self.__infer(image,domain_label)
        elif(tta_mode == 1): 
            # test time augmentation with flip in 2D
            # you may define your own method for test time augmentation
            outputs1 = self.__infer(image,domain_label=domain_label)
            outputs2 = self.__infer(torch.flip(image, [-2]),domain_label)
            outputs3 = self.__infer(torch.flip(image, [-1]),domain_label)
            outputs4 = self.__infer(torch.flip(image, [-2, -1]),domain_label)
            if(isinstance(outputs1, (tuple, list))):
                outputs = []
                for i in range(len(outputs1)):
                    temp_out1 = outputs1[i]
                    temp_out2 = torch.flip(outputs2[i], [-2])
                    temp_out3 = torch.flip(outputs3[i], [-1])
                    temp_out4 = torch.flip(outputs4[i], [-2, -1])
                    temp_mean = (temp_out1 + temp_out2 + temp_out3 + temp_out4) / 4
                    outputs.append(temp_mean)
            else:
                outputs2 = torch.flip(outputs2, [-2])
                outputs3 = torch.flip(outputs3, [-1])
                outputs4 = torch.flip(outputs4, [-2, -1])
                outputs = (outputs1 + outputs2 + outputs3 + outputs4) / 4
        else:
            raise ValueError("Undefined tta_mode {0:}".format(tta_mode))
        return outputs

