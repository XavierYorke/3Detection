#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/09/24 17:03:34
@Author  :   XavierYorke
@Contact :   mzlxavier1230@gmail.com
"""

import csv
import os
import pandas as pd
import json
"""
数据筛选：按标签体素排序后生成前n个数据的训练json文件
"""


if __name__ == '__main__':
    csv_path = r'saved/ias-S.csv'
    full_json = r'../environment/ias/ias.json'
    json_path = r'../environment/ias/ias-500.json'

    json_dict = {"training": [], "validation": []}
    with open(full_json, 'r') as f:
        full = json.load(f)
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        sorted_rows = sorted(reader, key=lambda m_row: int(m_row[1]), reverse=True)
        need = []
        for step, row in enumerate(sorted_rows, 1):
            need.append(row[0] + '/' + row[0] + '_origin.nii.gz')
            if step == 500:
                break

        step = 1
        for item in full['training']:
            if item['image'] in need:
                if step % 10 == 0:
                    json_dict['validation'].append(item)
                else:
                    json_dict['training'].append(item)
                step += 1
        for item in full['validation']:
            if item['image'] in need:
                if step % 10 == 0:
                    json_dict['validation'].append(item)
                else:
                    json_dict['training'].append(item)
                step += 1
    with open(json_path, "w") as outfile:
        json.dump(json_dict, outfile, indent=4)
