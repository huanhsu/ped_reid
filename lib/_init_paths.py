# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""Set up paths for tripletloss"""

import os.path as osp
import sys
import pdb

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
# Add caffe to PYTHONPATH
caffe_path = osp.join('/home/huan/person_search/caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir)
add_path(lib_path)
