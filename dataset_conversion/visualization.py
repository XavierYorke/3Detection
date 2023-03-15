import os
import SimpleITK as sitk
import json
import matplotlib.pyplot as plt


def visual(box, image_path):
    image = sitk.ReadImage(image_path)
    np_image = sitk.GetArrayFromImage(image)
    show_img = np_image[int((box[5] - box[2]) / 2)]
    box = [int(b) for b in box]
    print(box)
    show_img[box[0]: box[3]] = 3000
    show_img[box[1]: box[4]] = 3000
    plt.imshow(show_img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    json_root = r'../result/lung_image.json'
    with open(json_root) as f:
        data = json.load(f)['validation']
    for d in data:
        img_path = d['image']
        boxes = d['box']
        for box in boxes:
            visual(box, img_path)
            break
        break
