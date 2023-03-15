import json
import os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
from skimage import morphology, measure


def get_bounding_box(nii_path, image_name, expand_number=2):
    img = sitk.ReadImage(nii_path)
    img_arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    img_arr = np.transpose(img_arr, axes=(2, 1, 0))  # (x, y, z)
    img_arr[img_arr > 0] = 1
    off_arr, ele_arr = img.GetOrigin(), img.GetSpacing()
    # 去除其中小于10个像素的目标，防止噪声的干扰
    label = morphology.remove_small_objects(img_arr.astype(np.int16), min_size=4, connectivity=2)  # 2表示8邻接
    label = measure.label(label > 0)
    boxes = []
    labels = []
    for region in measure.regionprops(label):
        # 2d_bbox:(min_row, min_col, max_row, max_col)
        bbox = region.bbox
        r_x = int((bbox[3] - bbox[0] + expand_number)) * ele_arr[0]
        r_y = int((bbox[4] - bbox[1] + expand_number)) * ele_arr[1]
        r_z = int((bbox[5] - bbox[2] + expand_number)) * ele_arr[2]
        r = np.max([r_x, r_y, r_z])

        # 计算中心点坐标信息, 并进行处理以恢复到世界坐标系
        x = int((bbox[3] + bbox[0]) / 2) * ele_arr[0] + off_arr[0]
        y = int((bbox[4] + bbox[1]) / 2) * ele_arr[1] + off_arr[1]
        z = int((bbox[5] + bbox[2]) / 2) * ele_arr[2] + off_arr[2]

        box = [x, y, z, r, r, r]
        boxes.append(box)
        labels.append(0)
    img_info = {
        'box': boxes,
        'image': image_name,
        'label': labels
    }
    return img_info


if __name__ == '__main__':
    file_root = r'D:/Datasets/Lung/lung'
    save_path = r'../environment/Lung/lung.json'

    json_dict = {"training": [], "validation": []}
    files = sorted(os.listdir(file_root))
    train_len = int(0.9 * len(files))
    for step, file in tqdm(enumerate(files, 1)):
        origin_path = os.path.join(file_root, file, str(file) + '_origin.nii.gz')
        label_path = origin_path.replace('origin', 'lesion')
        file_name = file + '/' + file + '_origin.nii.gz'
        if os.path.exists(origin_path) and os.path.exists(label_path):
            data = get_bounding_box(label_path, file_name)
            if step <= train_len:
                json_dict['training'].append(data)
            else:
                json_dict['validation'].append(data)

    with open(save_path, "w") as outfile:
        json.dump(json_dict, outfile, indent=4)
