# crop ROIs
import os
import numpy as np
import SimpleITK as sitk


def read_nii(filepath):
    # 读取医学图像数据
    image = sitk.ReadImage(filepath)
    data = sitk.GetArrayFromImage(image)
    data = np.array(data, dtype='int16')
    return data


def save_nii_sitk(base_file, select_data, out_file):
    base_img = sitk.ReadImage(base_file)

    # origin = base_img.GetOrigin()
    direction = base_img.GetDirection()
    space = base_img.GetSpacing()
    save_img = sitk.GetImageFromArray(select_data)
    # save_img.SetOrigin(origin) #不保存原始的原点，以方便后续的操作
    save_img.SetDirection(direction)
    save_img.SetSpacing(space)
    sitk.WriteImage(save_img, out_file)


def crop_rois_thread(image_dir, files, out_base_dir, target='_lesion.nii.gz'):
    for f in files:
        origin_path = os.path.join(image_dir, f, str(f) + '_origin.nii.gz')
        lung_path = os.path.join(image_dir, f, str(f) + '_lungseg.nii.gz')
        label_path = os.path.join(image_dir, f, str(f) + target)
        if os.path.exists(origin_path) and os.path.exists(lung_path) and os.path.exists(label_path):
            lung_data = read_nii(lung_path)
            origin_data = read_nii(origin_path)
            label_data = read_nii(label_path)
            lung_data[lung_data > 0] = 1
            index = np.where(lung_data > 0)
            z_max, z_min = np.max(index[0]), np.min(index[0])
            y_max, y_min = np.max(index[1]), np.min(index[1])
            x_max, x_min = np.max(index[2]), np.min(index[2])
            select_origin = origin_data[z_min:z_max, y_min:y_max, x_min:x_max]
            select_label = label_data[z_min:z_max, y_min:y_max, x_min:x_max]
            select_label[select_label > 0] = 1
            out_file_dir = os.path.join(out_base_dir, f)
            if not os.path.exists(out_file_dir):
                os.makedirs(out_file_dir)
            out_origin_path = os.path.join(out_file_dir, str(f) + '_origin.nii.gz')
            out_label_path = os.path.join(out_file_dir, str(f) + target)

            # 进行数据的保存
            if not os.path.exists(out_origin_path):
                save_nii_sitk(base_file=origin_path, select_data=select_origin, out_file=out_origin_path)
            if not os.path.exists(out_label_path):
                save_nii_sitk(base_file=origin_path, select_data=select_label, out_file=out_label_path)
            print(f, 'completed')


if __name__ == "__main__":
    # from multiprocessing import cpu_count
    # from threading import Thread
    #
    # image_dir = r'D:\work_space_zy\dataset\lungdata'
    # out_base_dir = r'D:\work_space_zy\dataset\lession_crop'
    #
    #
    #
    # files = os.listdir(image_dir)
    #
    # thread_num = int(cpu_count() * 0.8)
    # thread_list = []
    # for index in range(thread_num):
    #     file_index = files[index:len(files):thread_num]
    #     work = Thread(target=crop_rois_thread, args=(image_dir, file_index, out_base_dir, ))
    #     work.start()
    #     thread_list.append(work)
    # for t in thread_list:
    #     t.join()
    base_dir = r'D:\Glx\Datasets\crop'
    files = os.listdir(base_dir)
    current_min = 10000
    for f in files:
        path = os.path.join(base_dir, f, f + '_origin.nii.gz')
        img = sitk.ReadImage(path)
        size = img.GetSize()
        a = np.array(size)
        print(f, size)
        if current_min > np.min(a):
            current_min = np.min(a)
    print(current_min)
    # path = r'D:\work_space_zy\dataset\lession_crop\lesion\crop_69123520201102_lesion.nii.gz'
    # img = sitk.ReadImage(path)
    # sitk.WriteImage(img, r'D:\Glx\Projects\3Detection\check.mhd')
