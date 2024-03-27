<!-- # FPL+
code for "FPL+: Filtered Pseudo Label-based Unsupervised Cross-Modality Adaptation for 3D Medical Image Segmentation"

The code is being gradually improved...

Train and test the FPL+:
```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:./PyMIC
python ./PyMIC/pymic/net_run_dsbn/net_run.py train ./config_dual/unet2d_dsbn_bst_t2s.cfg
python ./PyMIC/pymic/net_run_dsbn/net_run.py test ./config_dual/unet2d_dsbn_bst_t2s.cfg
``` -->


# FPL+: Filtered Pseudo Label-based Unsupervised Cross-Modality Adaptation for 3D Medical Image Segmentation
by [Jianghao Wu](https://jianghaowu.github.io/), et.al. 

## Introduction

This repository is for our paper **FPL+: Filtered Pseudo Label-based Unsupervised Cross-Modality Adaptation for 3D Medical Image Segmentation**. 


![](./FPL-plus.png)

## Data Preparation

### Dataset
[ Vestibular Schwannoma Segmentation Dataset](https://www.nature.com/articles/s41597-021-01064-w) | [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) | [MMWHS](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/)

For VS dataset, preprocess original data according to `./data/preprocess_vs.py`.

### Cross domian data augmentation 
Training [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and convert source domain data into source domian-like set and target domian-like set.

### File Organization
Using `./write_csv.py` to write your data into a `csv` file 

For vs data, hrT2 as source domain, ceT1 as target domain: 
``` 
├──hrT2
    ├── [train_source_like.csv]
        ├──image,label
        ├──/XX/img/vs_gk_192_t2.nii.gz,/XX/lab/vs_gk_192_t2.nii.gz
        ├──/XX/img/vs_gk_192_t2_S1.nii.gz,/XX/lab/vs_gk_192_t2.nii.gz
        ├──/XX/img/vs_gk_192_t2_S2.nii.gz,/XX/lab/vs_gk_192_t2.nii.gz
        ...
    ├── [train_target_like.csv]
        ├──image,label
        ├──/XX/img/vs_gk_192_t2_T1.nii.gz,/XX/lab/vs_gk_192_t2.nii.gz
        ├──/XX/img/vs_gk_192_t2_T2.nii.gz,/XX/lab/vs_gk_192_t2.nii.gz
        ...
    ├── [valid.csv]
        ├──image,label
        ├──/XX/img/vs_gk_165_t1.nii.gz,/XX/lab/vs_gk_165_t1.nii.gz
        ...
    ├── [test.csv]
        ├──image,label
        ├──/XX/img/vs_gk_10_t1.nii.gz,/XX/lab/vs_gk_10_t1.nii.gz
        ...
├──ceT1
    ├── [train.csv]
    ├── [valid.csv]
    ├── [test.csv]
```

## Training and Testing

### Train pseudo labels generator and get pseudo label
Write your training config file in `config_dual/unet2d_dsbn_vs_t2s.cfg`

```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:./PyMIC
python ./PyMIC/pymic/net_run_dsbn/net_run.py train ./config_dual/vs_t2s.cfg
python ./PyMIC/pymic/net_run_dsbn/net_run.py test ./config_dual/vs_t2s.cfg
```
### Train final segmentor
```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:./PyMIC
python ./PyMIC/pymic/net_run_dsbn/net_run.py train ./config_dual/vs_t2s_final_S.cfg
python ./PyMIC/pymic/net_run_dsbn/net_run.py test ./config_dual/vs_t2s_final_S.cfg
```


<!-- ## Acknowledgement
The U-Net model is borrowed from [Fed-DG](https://github.com/liuquande/FedDG-ELCFS). The Style Augmentation (SA) module is based on the nonlinear transformation in [Models Genesis](https://github.com/MrGiovanni/ModelsGenesis). The Dual-Normalizaiton is borrow from [DSBN](https://github.com/wgchang/DSBN). We thank all of them for their great contributions. -->

<!-- ## Citation

If you find this project useful for your research, please consider citing:

```bibtex
@inproceedings{zhou2022dn,
  title={Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization},
  author={Zhou, Ziqi and Qi, Lei and Yang, Xin and Ni, Dong and Shi, Yinghuan},
  booktitle={CVPR},
  year={2022}
}
``` -->
