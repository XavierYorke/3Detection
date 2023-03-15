#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/06/24 23:02:14
@Author  :   XavierYorke
@Contact :   mzlxavier1230@gmail.com
"""

import numpy as np
import os
import glob
import re
import time
import datetime
import math
from tqdm import tqdm, trange
import argparse


def modify_folders():
    folders = os.listdir(args.dir)
    for folder in tqdm(folders):
        folder = os.path.join(args.dir, folder)
        new_folder = folder[:-7]
        os.rename(folder, new_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modify file names in bulk')
    parser.add_argument('-d', '--dir', type=str, default='D:/Datasets/Aneurysm_resample', help='file dir')
    parser.add_argument('-o', '--output', type=str, default='', help='output dir')
    args = parser.parse_args()
    modify_folders()
