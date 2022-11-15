#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/09/24 17:03:34
@Author  :   XavierYorke
@Contact :   mzlxavier1230@gmail.com
'''

import csv
import os
import pandas as pd
import json


# 将csv文件转为LUNA16的json格式
def csv2json(csv_path, json_path):
    csv_file = pd.read_csv(csv_path, names=[0, 1, 2, 3, 4])
    files = sorted(list(set(csv_file[0].tolist())))
    aim_train = int(0.9 * len(files))
    suffix = '_origin.nii.gz'
    row0 = files[0]
    count = 0
    train = 0
    result_dict = {"training": [], "validation": []}
    result = {"box": []}
    for _, row in csv_file.iterrows():
        if row[0] != row0:
            result.update({"image": row0 + '/' + row0 + suffix})
            result.update({"label": [0] * count})
            if train < aim_train:
                result_dict["training"].append(result)
            else:
                result_dict["validation"].append(result)
            count = 0
            row0 = row[0]
            train += 1
            result = {"box": []}
        result["box"].append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[4]), float(row[4])])
        count += 1
    result.update({"image": row0 + '/' + row0 + suffix})
    result.update({"label": [0] * count})
    result_dict["validation"].append(result)
    with open(json_path, "w") as outfile:
        json.dump(result_dict, outfile, indent=4)


if __name__ == '__main__':
    file_name = 'lung'
    csv_path = './files/' + file_name + '.csv'
    json_path = '../config/' + file_name + '.json'
    csv2json(csv_path, json_path)