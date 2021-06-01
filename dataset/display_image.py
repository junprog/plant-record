import os
import glob
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Display datas')
    parser.add_argument('--data-dir', default='C:/Users/junya/Documents/plant_segmentation_data',
                        help='dataset directory')
    args = parser.parse_args()
    return args

def display(display_list):
    plt.figure(figsize=(10, 8))

    title = ['Input Image', 'True Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')

    plt.show()

if __name__ == '__main__':
    args = parse_args()

    for phase in ['train', 'test']:
        data_dir = os.path.join(args.data_dir, phase)
        img_list = glob.glob(os.path.join(data_dir, '*.jpg'))

        for img_path in img_list:
            img = Image.open(img_path)
            print(img_path, img.size)
            gt_path = img_path.replace('.jpg', '_mask.png')
            gt = np.array(Image.open(gt_path))

            display([img, gt])
