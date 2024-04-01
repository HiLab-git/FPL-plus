import SimpleITK as sitk
import numpy as np
import os

pseudo_target_root = './results_dual/vs_t1s_g_i-train_hrT2'
pseudo_fake_source_root = './results_dual/vs_t1s_g_i-train_hrT2-ceT1_cyc'
t2s = os.listdir(pseudo_target_root)
t2_cycs = os.listdir(pseudo_fake_source_root)
t2_names = [item for item in t2s if '.nii.gz' in item]
t2_cyc_names = [item for item in t2_cycs if '.nii.gz' in item]
assert len(t2_names) == len(t2_cyc_names)
for name in t2_names:
    t2_full = os.path.join(pseudo_target_root,name)
    t2_cyc_full = os.path.join(pseudo_fake_source_root,name)
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
    sub_rev = np.ones_like(sub_arr)
    sub_rev = sub_rev-sub_arr*0.5
    sub_rev = sitk.GetImageFromArray(sub_rev)
    sitk.WriteImage(sub_rev,'./dataset/hrT2_pixel-weight/'+name)
  