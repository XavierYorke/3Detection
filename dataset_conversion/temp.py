import os

if __name__ == '__main__':
    root_path = r'D:/Datasets/Lung/JiangNan_resample'
    orig_json = open(r'../config/JiangNan_resample.json', 'w')

    file_list = os.listdir(root_path)
    orig_json.write('{"validation": [')
    count = 0
    for step, file in enumerate(file_list, 1):
        orig_path = os.path.join(file)
        orig_json.write('{"image": "')
        orig_json.write(file + '/' + file + '.nii.gz')
        orig_json.write('"},')
        # count += 1
        # if count == 100:
        #     break
    orig_json.write(']}')
    orig_json.close()


