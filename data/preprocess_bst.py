import csv
import os
from posixpath import split 
import SimpleITK as sitk 
import numpy as np 
def winadj_mri(array):
    v0 = np.percentile(array, 1)
    v1 = np.percentile(array, 999)
    array[array < v0] = v0    
    array[array > v1] = v1  
    v0 = array.min() 
    v1 = array.max() 
    array = (array - v0) / (v1 - v0) * 2.0 - 1.0
    return array
def crop_depth(img,lab):
    D,W,H = img.shape
    indices = np.where(lab>0)
    d00, d11 = indices[0].min(), indices[0].max()
    zero_img = img[max(d00-16,0):min(d11+16,D),:,:]
    zero_lab = lab[max(d00-16,0):min(d11+16,D),:,:]
    return zero_img,zero_lab

if __name__ == "__main__":
    ref_moda = 'flair'
    moda = 'flair'
    phase = 'train'
    roots = '/your/MICCAI_BraTS2020_TrainingData'
    img_path = '/your/bst_data/'+ref_moda+'_'+phase+'/img'
    save_img = '/your/bst_data/'+moda+'_'+phase+'/img'
    save_lab = '/your/bst_data/'+moda+'_'+phase+'/lab'
    if(not os.path.exists(save_lab)):
        os.mkdir(save_lab)
    names = os.listdir(img_path)

    for i in names:
        caase_name = i.split('.nii')[0]
        print(i,caase_name)
        case_root = os.path.join(roots,caase_name)
        img_obj_ = sitk.ReadImage(case_root + '/' + caase_name +'_'+ moda+'.nii.gz')
        lab_obj_ = sitk.ReadImage(case_root + '/' + caase_name +'_seg'+'.nii.gz')
        lab_obj = sitk.GetArrayFromImage(lab_obj_)
        img_obj = sitk.GetArrayFromImage(img_obj_)
        lab_obj[lab_obj>0] = 1
        img_obj,lab_obj = crop_depth(img_obj,lab_obj)
        lab_obj = sitk.GetImageFromArray(lab_obj)
        img_obj = sitk.GetImageFromArray(img_obj)
        img_save_dir = os.path.join(save_img,caase_name+'.nii.gz')
        lab_save_dir = os.path.join(save_lab,caase_name+'.nii.gz')
        sitk.WriteImage(img_obj, img_save_dir)
        sitk.WriteImage(lab_obj, lab_save_dir)