import os
import csv
import SimpleITK as sitk
import nibabel as nib
"""
用于数据分析
"""


def get_spacing(file_path):
    image = sitk.ReadImage(file_path)
    spacing = image.GetSpacing()
    print(spacing)


def get_size(file_path):
    img = nib.load(file_path)
    return img.shape


if __name__ == '__main__':
    root = 'D:/Datasets/CADA/CADA-resample'
    file_list = os.listdir(root)
    file_list = [root + '/' + file + '/' + file + '.nii.gz' for file in file_list]
    with open('saved/CADA_shape.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'x', 'y', 'z'])
        for file in file_list:
            shape = get_size(file)
            data = [file] + [shape[0]] + [shape[1]] + [shape[2]]
            writer.writerow(data)
