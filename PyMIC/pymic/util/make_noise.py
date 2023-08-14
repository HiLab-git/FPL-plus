import numpy as np
import torch
import cv2
from skimage import measure, draw
# ==================================================================
# ==================================================================        
def make_roi_mask(labels,
                  roi_type = 'fg_only'):
    
    roi_mask = np.zeros_like(labels, dtype = np.float32)

    # ==========================    
    # for each image in the batch
    # ==========================
    for i in range(roi_mask.shape[0]):
        
        # ==========================
        # find the boundbox for this image
        # ==========================
        fg_indices = np.array(np.where(labels[i,:,:] != 0))
        
        # ==========================
        # highlight the fg pixels, if there are fg pixels in this image
        # else highlight the entire image
        # ==========================
        if fg_indices.shape[1] != 0:

            # highlight the bounding box if there are fg pixels in this image            
            # n = 25 # number of extra pixels outside the fg labels
            # roi_mask[i,
            # np.maximum(np.min(fg_indices[0,:])-n, 0) : np.minimum(np.max(fg_indices[0,:])+n, labels.shape[1]),
            # np.maximum(np.min(fg_indices[1,:])-n, 0) : np.minimum(np.max(fg_indices[1,:])+n, labels.shape[2])] = 1.0

            # highlight exactly those pixels which have non-zero label predictions
            roi_mask[i,
                     fg_indices[0,:],
                     fg_indices[1,:]] = 1.0
        else:
             roi_mask[i, :, :] = 1.0
    
    # in this case, provide the roi as the entire image.
    # this is used while training the l2i mapper.
    if roi_type is 'entire_image':
        roi_mask = np.ones_like(labels, dtype = np.float32)
    
    return np.expand_dims(roi_mask, axis=-1)

# ==================================================================
# ==================================================================
def make_noise_masks_2d(shape,
                        mask_type,
                        mask_params,
                        is_num_masks_fixed,
                        is_size_masks_fixed,
                        nlabels,
                        labels_1hot = None):
    
    blank_masks = np.ones(shape = shape)
    wrong_labels = np.zeros(shape = shape)

    # ====================        
    # for each image in the batch
    # ====================
    for i in range(shape[0]):
            
        # ====================
        # make a random number of noise boxes in this image
        # ====================
        if is_num_masks_fixed is True:
            num_noise_squares = mask_params[1]
        else:
            num_noise_squares = np.random.randint(1, mask_params[1]+1)
            
        for _ in range(num_noise_squares):
                
            # ====================
            # choose the size of the noise box randomly 
            # ====================
            if is_size_masks_fixed is True:
                r = mask_params[0]
            else:
                r = np.random.randint(1, mask_params[0]+1)
                
            # ====================
            # choose the center of the noise box randomly 
            # ====================
            mcx = np.random.randint(r+1, shape[1]-r-1)
            mcy = np.random.randint(r+1, shape[2]-r-1)
                
            # ====================
            # set the labels in this box to 0
            # ====================
            blank_masks[i, mcx-r:mcx+r, mcy-r:mcy+r, :] = 0

            if mask_type is 'random':                
                # ====================
                # set the labels in this box to an arbitrary label
                # ====================
                wrong_labels[i, mcx-r:mcx+r, mcy-r:mcy+r, np.random.randint(nlabels)] = 1
            
            elif mask_type is 'jigsaw':               
                # ====================
                # choose another box in the image from which copy labels to the previous box
                # ====================
                mcx_src = np.random.randint(r+1, shape[1]-r-1)
                mcy_src = np.random.randint(r+1, shape[2]-r-1)
                wrong_labels[i, mcx-r:mcx+r, mcy-r:mcy+r, :] = labels_1hot[i, mcx_src-r:mcx_src+r, mcy_src-r:mcy_src+r, :]
                
            elif mask_type is 'zeros':
                # ====================                
                # set the labels in this box to zero
                # ====================
                wrong_labels[i, mcx-r:mcx+r, mcy-r:mcy+r, 0] = 1
        
    return blank_masks, wrong_labels

# ==================================================================
# ==================================================================
def make_noise_masks_3d(shape,
                        mask_type,
                        mask_params,
                        nlabels,
                        labels_1hot = None,
                        is_num_masks_fixed = False,
                        is_size_masks_fixed = False):
    
    blank_masks = np.ones(shape = shape)
    wrong_labels = np.zeros(shape = shape)
                   
    # ====================
    # make a random number of noise boxes in this (3d) image
    # ====================
    if is_num_masks_fixed is True:
        num_noise_squares = mask_params[1]
    else:
        num_noise_squares = np.random.randint(1, mask_params[1]+1)
        
    for _ in range(num_noise_squares):
            
        # ====================
        # choose the size of the noise box randomly 
        # ====================
        if is_size_masks_fixed is True:
            r = mask_params[0]
        else:
            r = np.random.randint(1, mask_params[0]+1)
            
        # choose the center of the noise box randomly 
        mcx = np.random.randint(r+1, shape[1]-r-1)
        mcy = np.random.randint(r+1, shape[2]-r-1)
        mcz = np.random.randint(r+1, shape[3]-r-1)
            
        # set the labels in this box to 0
        blank_masks[:, mcx-r:mcx+r, mcy-r:mcy+r, mcz-r:mcz+r, :] = 0
        
        if mask_type is 'squares_jigsaw':               
            # choose another box in the image from which copy labels to the previous box
            mcx_src = np.random.randint(r+1, shape[1]-r-1)
            mcy_src = np.random.randint(r+1, shape[2]-r-1)
            mcz_src = np.random.randint(r+1, shape[3]-r-1)
            wrong_labels[:, mcx-r:mcx+r, mcy-r:mcy+r, mcz-r:mcz+r, :] = labels_1hot[:, mcx_src-r:mcx_src+r, mcy_src-r:mcy_src+r, mcz_src-r:mcz_src+r, :]
            
        elif mask_type is 'squares_zeros':                
            # set the labels in this box to zero
            wrong_labels[:, mcx-r:mcx+r, mcy-r:mcy+r, mcz-r:mcz+r, 0] = 1
    
    return blank_masks, wrong_labels

# def make_noise_masks_3d(lab,patch_size = [3,5,5],patch_num = 10):
#     lab = lab.cpu().numpy()
#     # print(lab[:,1].sum(),'169-169',lab.shape)
#     # print(lab[:,0].sum(),'169-169')
#     if len(lab.shape) == 4:
#         c,w,h,d = lab.shape
#     elif len(lab.shape) == 5:
#         n,c,w,h,d = lab.shape
#     for i in range(patch_num):
#         tt = np.random.randint(0, 5)
#         if tt == 0:
#             w1 = np.random.randint(0, w - patch_size[0])
#             h1 = np.random.randint(0, h - patch_size[1])
#             d1 = np.random.randint(0, d - patch_size[2])
#         elif tt > 0:
#             indices = np.where(lab[0,1]>0)
#             # print(indices[0])
#             try:
#                 d00, d11 = indices[0].min(), indices[0].max()
#                 h00, h11 = indices[1].min(), indices[1].max()
#                 w00, w11 = indices[2].min(), indices[2].max()
                
#                 w1 = np.random.randint(max(d00-4,0), min(d11+4,w-patch_size[0]))
#                 h1 = np.random.randint(max(h00-5,0), min(h11+5,h-patch_size[1]))
#                 d1 = np.random.randint(max(w00-5,0), min(w11+5,d-patch_size[2]))
#             except:
#                 w1 = np.random.randint(0, w - patch_size[0])
#                 h1 = np.random.randint(0, h - patch_size[1])
#                 d1 = np.random.randint(0, d - patch_size[2])

#         patch_0 = lab[:,0:1,w1:w1 + patch_size[0], h1:h1 + patch_size[1], d1:d1 + patch_size[2]]
#         patch_1 = lab[:,1:2,w1:w1 + patch_size[0], h1:h1 + patch_size[1], d1:d1 + patch_size[2]]
#         # lab[:,0,w1:w1 + patch_size[0], h1:h1 + patch_size[1], d1:d1 + patch_size[2]] = patch_1
#         # lab[:,1,w1:w1 + patch_size[0], h1:h1 + patch_size[1], d1:d1 + patch_size[2]] = patch_0
#         lab[:,0:1,w1:w1 + patch_size[0], h1:h1 + patch_size[1], d1:d1 + patch_size[2]] = np.ones_like(patch_0)-patch_0
#         lab[:,1:2,w1:w1 + patch_size[0], h1:h1 + patch_size[1], d1:d1 + patch_size[2]] = np.ones_like(patch_1)-patch_1
#         # print(i,patch_1.sum(),patch_0.sum(),patch_0.shape,patch_1.shape,'sum')
#     # print(lab[:,1].sum(),'204-204')
#     return torch.from_numpy(lab).cuda()
def conv3d_data2(data,k_size):#k_size[c,h,w]
    padd_c = k_size[0]//2
    padd_h = k_size[1]//2
    padd_w = k_size[2]//2
    x_pad = np.pad(data,((padd_c,padd_c),(padd_h,padd_h),(padd_w,padd_w)))
    c_num = data.shape[0]
    h_num = data.shape[1]
    w_num = data.shape[2]
    temp_c = np.array([x_pad[i:i+k_size[0],:,:] for i in range(c_num)])
    temp_h = np.array([temp_c[:,:,i:i+k_size[1],:] for i in range(h_num)])
    temp_w = np.array([temp_h]*k_size[2]).transpose([2,1,0,3,4,5])
    for i in range(1,k_size[2]):
        temp_w[:,:,i,:,:,:-i] = temp_w[:,:,i,:,:,i:]
    result = temp_w[:,:,:,:,:,:-(k_size[2]-1)].transpose([0,1,5,3,4,2])
    return result 
from scipy import ndimage
def get_largest_component(image):
    """
    get the largest component from 2D or 3D binary image
    image: nd array
    """
    dim = len(image.shape)
    # print(dim,image.shape)
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return  output
def erode_rect3d(img,K_size):
    data = conv3d_data2(img>0,K_size)
    K = np.ones(K_size)
    summ = K.shape[0]*K.shape[1]*K.shape[2]
    result = np.einsum("abcdef,def->abc",data,K)
    result = (result==summ)*1.0
    return result
def dilate_rect3d(img,K_size):
    data = conv3d_data2(img>0,K_size)
    K = np.ones(K_size)
    result = np.einsum("abcdef,def->abc",data,K)
    result = (result>0)*1.0
    return result

def make_noise_masks_3d(lab,patch_size = [6,16,16],patch_num = 5):
    patch_num = np.random.randint(0, patch_num)
    lab = lab.cpu().numpy()
    lab_ = lab
    lab = get_largest_component(lab[0,0])
    if lab.sum() <20:
        return torch.from_numpy(lab_).int().cuda()
    else:
        try:
            for i in range(patch_num):
                indices = np.where(lab>0)
                d00, d11 = indices[0].min(), indices[0].max()
                d1 = np.random.randint(d00, d11)
                indices = np.where(lab[d1]>0)
                w00, w11 = indices[0].min(), indices[0].max() 
                w1 = np.random.randint(w00, w11)
                indices = np.where(lab[d1,w1]>0)
                h00, h11 = indices[0].min(), indices[0].max() 
                patch_0 = lab[int(d1-(patch_size[0]/2)):int(d1+(patch_size[0]/2)), int(w1-(patch_size[1])/2):int(w1+(patch_size[1]/2)), int(h00-(patch_size[2]/2)):int((h00+(patch_size[2]/2)))]
                patch_1 = lab[int(d1-patch_size[0]/2):int(d1+patch_size[0]/2), int(w1-patch_size[1]/2):int(w1+patch_size[1]/2), int(h11-patch_size[2]/2):int(h11+patch_size[2]/2)]
                tt = np.random.randint(0, 2)
                if tt == 0:
                    patch_0 = erode_rect3d(patch_0,K_size=[3,3,3])
                    patch_1 = erode_rect3d(patch_1,K_size=[3,3,3])
                else:
                    patch_0 = dilate_rect3d(patch_0,K_size=[3,3,3])
                    patch_1 = dilate_rect3d(patch_1,K_size=[3,3,3])
                lab_[0,0,int(d1-(patch_size[0]/2)):int(d1+(patch_size[0]/2)), int(w1-(patch_size[1])/2):int(w1+(patch_size[1]/2)), int(h00-(patch_size[2]/2)):int((h00+(patch_size[2]/2)))] = patch_0
                lab_[0,0,int(d1-patch_size[0]/2):int(d1+patch_size[0]/2), int(w1-patch_size[1]/2):int(w1+patch_size[1]/2), int(h11-patch_size[2]/2):int(h11+patch_size[2]/2)] = patch_1
            lab_ = torch.from_numpy(lab_).int().cuda()
            return lab_
        except:
            return torch.from_numpy(lab_).int().cuda()