export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:./PyMIC
python ./PyMIC/pymic/net_run_dsbn/net_run.py train ./config_dual/unet2d_dsbn_bst_t2s.cfg
python ./PyMIC/pymic/net_run_dsbn/net_run.py test ./config_dual/unet2d_dsbn_bst_t2s.cfg


