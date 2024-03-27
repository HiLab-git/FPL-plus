import os 
import SimpleITK as sitk 
import numpy as np 
from scipy import ndimage 

def get_source_image_info():
    img_dir = "./input/"
    img_names = os.listdir(img_dir)
    lab_names = [item for item in img_names if "Label.nii.gz" in item]
    lab_names = sorted(lab_names)
    num = len(lab_names)
    print("image number", num)

    dmin_list, dmax_list = [], []
    hmin_list, hmax_list = [], []
    wmin_list, wmax_list = [], []
    for lab_name in lab_names:
        lab_obj = sitk.ReadImage(img_dir + '/' + lab_name)
        lab = sitk.GetArrayFromImage(lab_obj)
        spacing = lab_obj.GetSpacing()
        [D, H, W] = lab.shape 

        # get upper and lower bound of labels
        indices = np.where(lab>0)
        d0, d1 = D - indices[0].max(), D - indices[0].min()
        dp0, dp1 = d0 * spacing[2], d1 * spacing[2]
        dmin_list.append(dp0)
        dmax_list.append(dp1)

        h0, h1 = indices[1].min(), indices[1].max()
        w0, w1 = indices[2].min(), indices[2].max()
        hmin_list.append(h0)
        hmax_list.append(h1)
        wmin_list.append(w0)
        wmax_list.append(w1)
        print(lab_name, D, H, W, d0, d1, dp0, dp1)
    dmin_list, dmax_list = np.asarray(dmin_list), np.asarray(dmax_list)
    hmin_list, hmax_list = np.asarray(hmin_list), np.asarray(hmax_list)
    wmin_list, wmax_list = np.asarray(wmin_list), np.asarray(wmax_list)
    print("upper bound, min, mean, max", dmin_list.min(), dmin_list.mean(), dmin_list.max())
    print("lower bound, min, mean, max", dmax_list.min(), dmax_list.mean(), dmax_list.max())
    print("inferior bound,  min, mean, max", hmin_list.min(), hmin_list.mean(), hmin_list.max())
    print("posterior bound, min, mean, max", hmax_list.min(), hmax_list.mean(), hmax_list.max())
    print("left bound,  min, mean, max", wmin_list.min(), wmin_list.mean(), wmin_list.max())
    print("right bound, min, mean, max", wmax_list.min(), wmax_list.mean(), wmax_list.max())

def get_target_image_info():
    img_dir = "./input/"
    img_names = os.listdir(img_dir)
    img_names = sorted(img_names)
    num = len(img_names)
    print("image number", num)

    for img_name in img_names:
        img_obj = sitk.ReadImage(img_dir + '/' + img_name)
        img = sitk.GetArrayFromImage(img_obj)
        spacing = img_obj.GetSpacing()
        [D, H, W] = img.shape
        img_id = img_name.split('_')[1] 

def source_image_crop():
    img_dir = "/your/VS/source/dir"
    out_dir = "/your/vs/source/save_dir"
    img_names = os.listdir(img_dir)
    img_names = [item for item in img_names if "t1" in item]
    for img_name in img_names:
        lab_name = img_name.replace("ceT1", "Label")
        img_obj = sitk.ReadImage(img_dir + '/' + img_name)
        lab_obj = sitk.ReadImage(img_dir + '/' + lab_name)
        img = sitk.GetArrayFromImage(img_obj)
        lab = sitk.GetArrayFromImage(lab_obj)

        [D, H, W] = img.shape
        spacing = img_obj.GetSpacing()
        # crop image based on predefined bounding box
        d0 = int(D - 153/spacing[2])
        d1 = int(D - 93/spacing[2]) 
        h0, h1 = 190, 350 
        w0, w1 = 120, 392 
        img_sub = img[d0:d1, h0:h1, w0:w1]
        lab_sub = lab[d0:d1, h0:h1, w0:w1]
        assert(lab_sub.sum() == lab.sum())

        #convert array to image object
        out_img_obj = sitk.GetImageFromArray(img_sub)
        out_lab_obj = sitk.GetImageFromArray(lab_sub)
        direct = img_obj.GetDirection()
        origin = img_obj.GetOrigin()
        out_img_obj.SetSpacing(spacing)
        out_img_obj.SetDirection(direct)
        out_img_obj.SetOrigin(origin)
        sitk.WriteImage(out_img_obj, out_dir + '/' + img_name)

        out_lab_obj.CopyInformation(out_img_obj)
        sitk.WriteImage(out_lab_obj, out_dir + '/' + lab_name)

def target_image_crop():
    img_dir = "/your/VS/dir"
    out_dir = "/your/vs/save_dir"
    img_names = os.listdir(img_dir)
    img_names = [item for item in img_names if "t2.nii.gz" in item]
    for img_name in img_names:
        img_obj = sitk.ReadImage(img_dir + '/' + img_name)
        img = sitk.GetArrayFromImage(img_obj)
        [D, H, W] = img.shape
        spacing = img_obj.GetSpacing()
        if(D < 50):
            d0, d1 = 5, D-5
        elif(spacing[2] == 1.0):
            d0, d1 = 8, 48 
        elif(spacing[2] == 1.5):
            d0, d1 = 8, 48 
        else:
            raise ValueError("undefined case")
        
        # crop image based on predefined bounding box
        h0, h1 = 120, 376
        w0, w1 = 120, 376
        h0, h1 = int(h0*H/512), int(h1*H/512)
        w0, w1 = int(w0*W/512), int(w1*W/512)
        img_sub = img[d0:d1, h0:h1, w0:w1]

        # resample to 160x272
        [Ds, Hs, Ws] = img_sub.shape
        zoom = [1.0, 256.0/Hs, 256.0/Ws]
        img_sub = ndimage.zoom(img_sub, zoom)
        #convert array to image object
        out_img_obj = sitk.GetImageFromArray(img_sub)
    
        direct = img_obj.GetDirection()
        origin = img_obj.GetOrigin()
        out_img_obj.SetSpacing([0.4102, 0.4102, spacing[2]])
        out_img_obj.SetDirection(direct)
        out_img_obj.SetOrigin(origin)
        sitk.WriteImage(out_img_obj, out_dir + '/' + img_name)
       
if __name__ == "__main__":
    # get_source_image_info()
    source_image_crop()
    # get_target_image_info()
    target_image_crop()