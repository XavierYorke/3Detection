#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/09/24 17:03:34
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

import csv
import os


# 将csv文件转为LUNA16的json格式
def csv2json(csv_path, json_path):
    csv_file = open(csv_path, 'r')
    csv_reader = csv.reader(csv_file)
    json_file = open(json_path, 'w')
    json_file.write('[')
    pre_row = 'temp'
    count_L = -1
    flag = 1
    for row in csv_reader:
        if flag:
            json_file.write('{')
            json_file.write('"box": [')
            flag = 0

        if pre_row == 'temp':
            pre_row = row[0]

        json_file.write('[' + row[1] + ',' + row[2] + ',' +
                        row[3] + ',' + row[4] + ',' + row[4] + ',' + row[4] + '],')
        count_L += 1
        if row[0] != pre_row:
            json_file.close()
            with open(json_path, "rb+") as f:

                f.seek(-1, os.SEEK_END)
                f.truncate()
                f.close()
            json_file = open(json_path, 'a+')
            json_file.write('],')

            json_file.write(
                # '"image": "' + row[0] + '/' + row[0] + '_origin.nii.gz' + '",')
                '"image": "' + row[0] + '/' + row[0] + '.nii.gz' + '",')
            json_file.write('"label": [')
            for _ in range(count_L):
                json_file.write('0,')
            json_file.write('0]')
            json_file.write('}')
            json_file.write(',')
            count_L = -1
            pre_row = row[0]
            flag = 1

    json_file.write('],')
    json_file.write(
        # '"image": "' + row[0] + '/' + row[0] + '_origin.nii.gz' + '",')
        '"image": "' + row[0] + '/' + row[0] + '.nii.gz' + '",')
    json_file.write('"label": [')
    for _ in range(count_L):
        json_file.write('0,')
    json_file.write('0]')
    json_file.write('}')
    json_file.write(',')
    count_L = -1
    pre_row = row[0]

    json_file.write(']')
    csv_file.close()
    json_file.close()


if __name__ == '__main__':
    file_name = 'lung_100'
    csv_path = './files/' + file_name + '.csv'
    json_path = '../config/' + file_name + '.json'
    csv2json(csv_path, json_path)
