import os

if __name__ == '__main__':
    root_path = r'D:/Datasets/Lung/lungdata'
    orig_json = open(r'./files/lung_100_origin.json', 'w')
    label_json = open(r'./files/lung_100_lesion.json', 'w')

    file_list = os.listdir(root_path)
    orig_json.write('{"training": [')
    label_json.write('{"training": [')
    count = 0
    for step, file in enumerate(file_list, 1):
        orig_path = os.path.join(file, file + '_origin.nii.gz')
        label_path = os.path.join(file, file + '_lesion.nii.gz')
        if os.path.exists(os.path.join(root_path, orig_path)) and os.path.exists(os.path.join(root_path, label_path)):
            orig_json.write('{"image": "')
            orig_json.write(file + '/' + file + '_origin.nii.gz')
            orig_json.write('"},')
            label_json.write('{"image": "')
            label_json.write(file + '/' + file + '_lesion.nii.gz')
            label_json.write('"},')
            count += 1
        if count == 100:
            break
    orig_json.write(']}')
    label_json.write(']}')
    orig_json.close()
    label_json.close()


