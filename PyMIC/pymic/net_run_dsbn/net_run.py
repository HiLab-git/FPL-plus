# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import os
import sys
from pymic.util.parse_config import *
from pymic.net_run_dsbn.agent_cls import ClassificationAgent
from pymic.net_run_dsbn.agent_seg import SegmentationAgent
from pymic.util.evaluation_seg_train import eva_main

def main():
    """
    The main function for running a network for training or inference.
    """
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('   pymic_run train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    log_dir  = config['training']['ckpt_save_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(stage), level=logging.INFO,
                        format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)
    task     = config['dataset']['task_type']
    assert task in ['cls', 'cls_nexcl', 'seg']
    if(task == 'cls' or task == 'cls_nexcl'):
        agent = ClassificationAgent(config, stage)
    else:
        agent = SegmentationAgent(config, stage)
    agent.run()
    if stage != 'test':
        agent2 = SegmentationAgent(config, 'test')
        agent2.run()
    eva_main(config)

if __name__ == "__main__":
    main()
    

