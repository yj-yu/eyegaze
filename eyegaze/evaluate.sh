#!/bin/bash

export LD_PRELOAD="/usr/lib/libtcmalloc.so.4"
python evaluate.py --checkpoint_dir='/data1/yj/experiment/cvs_test1/'
