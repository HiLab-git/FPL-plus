from scipy import ndimage
import SimpleITK as sitk
import numpy as np
import os

t2_root = '/data2/jianghao/VS/vs_seg2021/results_bst/dsbn_t2-f_12+t2_0_flair_train'
t2_cyc_root = '/data2/jianghao/VS/vs_seg2021/results_bst/dsbn_t2-f_12+t2_0_f-t2_train_cyc'
t2s = os.listdir(t2_root)
t2_cycs = os.listdir(t2_cyc_root)
t2_names = [item for item in t2s if '.nii.gz' in item]
t2_cyc_names = [item for item in t2_cycs if '.nii.gz' in item]
assert len(t2_names) == len(t2_cyc_names)
for name in t2_names:
    t2_full = os.path.join(t2_root,name)
    t2_cyc_full = os.path.join(t2_cyc_root,name)
    t2_full = sitk.ReadImage(t2_full)
    t2_cyc_full = sitk.ReadImage(t2_cyc_full)
    t2_full = sitk.GetArrayFromImage(t2_full)
    t2_cyc_full = sitk.GetArrayFromImage(t2_cyc_full)
    assert t2_full.shape == t2_cyc_full.shape
    # assert t2_cyc_full.max() == t2_full.max()
    both_arr = t2_full+t2_cyc_full
    both_arr[both_arr > 1] = 1
    and_arr = t2_cyc_full*t2_full
    sub_arr = both_arr - and_arr
    print(both_arr.sum(),and_arr.sum(),sub_arr.sum(),sub_arr.max())
    sub_rev = np.ones_like(sub_arr)
    sub_rev = sub_rev-sub_arr*0.5
    sub_rev = sitk.GetImageFromArray(sub_rev)
    sitk.WriteImage(sub_rev,'/data2/jianghao/VS/vs_seg2021/script/FPL-UDA/bst_t2s_sub_arr/'+name)
  