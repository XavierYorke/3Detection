import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import numpy as np
from collections import Counter
import csv


# 读取图像
def read_medical_image(filename):
    image = sitk.ReadImage(filename)
    array = sitk.GetArrayFromImage(image)
    return np.float32(array), array.shape


# 提取标签
def extract_interesting_regions(root, origin, label):
    basename = os.path.basename(root)
    ct_array, ct_shape = read_medical_image(os.path.join(root, basename + origin))
    seg_array, _ = read_medical_image(os.path.join(root, basename + label))
    seg_mask = seg_array == 1
    return ct_array[np.where(seg_mask > 0)], ct_shape


# 分布可视化
def distribution_show(img):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.distplot(img, color='red', label="label", ax=ax)
    new_ticks = np.linspace(-500, 1500, 21)
    plt.xticks(new_ticks)
    plt.legend()
    plt.savefig(save_name)
    plt.show()


# 统计标签的体素，强度范围以及数据shape
# result用于查看所有图像的标签强度分布情况，返回后用上面的可视化函数生成结果
def main(base_root, origin, label):
    # result = []
    for root, dirs, files in os.walk(base_root):
        for step, _dir in tqdm(enumerate(dirs)):
            _file_root = os.path.join(root, _dir)
            ct_aim, ct_shape = extract_interesting_regions(_file_root, origin, label)
            ct_voxel = ct_aim.shape[0]
            # result.append(ct_aim)
            row = [_dir, ct_voxel, min(ct_aim), max(ct_aim), ct_aim.mean(), ct_shape[::-1]]
            csv_writer.writerow(row)
            # if step == 2:
            #     break
    # return np.concatenate(result, axis=0)


if __name__ == '__main__':
    file_root = 'D:/Datasets/Aneurysm/S/one/'
    origin_postfix = '_origin.nii.gz'
    label_postfix = '_ias.nii.gz'
    save_name = 'saved/ias-A.png'
    csv_path = 'saved/ias-S.csv'
    fid = open(csv_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(fid)
    csv_writer.writerow(['file', 'voxel', 'min', 'max', 'mean', 'shape'])

    main(file_root, origin_postfix, label_postfix)
    fid.close()


