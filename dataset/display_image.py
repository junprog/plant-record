import os
import glob
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Display datas')
    parser.add_argument('--data-dir', default='D:/Users/junya/Documents/plant_segmentation_data',
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

        #compare_list = list(range(1,42))

        for img_path in img_list:
            img = Image.open(img_path)
            print(img_path, img.size)
            gt_path = img_path.replace('.jpg', '.png')
            gt = np.array(Image.open(gt_path))

            """
            回転処理
            if str.isnumeric(os.path.basename(img_path).replace('.jpg', '')):
                if int(os.path.basename(img_path).replace('.jpg', '')) and set(compare_list):
                    
                    img = img.rotate(270, expand=True)
                    img.save(img_path)
            """
            
            display([img, gt])
