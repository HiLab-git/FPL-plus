# -*- coding: utf-8 -*-
from __future__ import print_function, division
import sys
from pymic.net_run_nll.nll_clslsr import get_confidence_map 


if __name__ == "__main__":
    """
    The main function to get the confidence map during inference.
    """
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('   python nll_clslsr.py config.cfg')
        exit()
    cfg_file = str(sys.argv[1])
    get_confidence_map(cfg_file)