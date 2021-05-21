"""
画像の短辺を 256 [px] とした等方スケール変換を行う。
(e.g.) 1000 x 2000 -> 256 x 512
"""

import os
import glob

import cv2
import numpy as np
import argparse

MIN_EDGE = 256

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='',
                        help='original data directory')
    parser.add_argument('--save-dir', default='',
                        help='save data directory')
    args = parser.parse_args()
    return args

def scaling_img(img):
    min_axis = np.argmin(img.shape[:2])
    h, w = img.shape[:2]
    new_edge = round(w * (MIN_EDGE / h))

    if min_axis == 0:
        out = cv2.resize(img, dsize=(new_edge, MIN_EDGE))
    elif min_axis == 1:
        out = cv2.resize(img, dsize=(MIN_EDGE, new_edge))

    return out

if __name__ == '__main__':
    args = parse_args()
    origin_dir = args.origin_dir
    save_dir = args.save_dir
    
    origin_img_path_list =  glob.glob(os.path.join(origin_dir, '*.jpg'))

    for img_path in origin_img_path_list:
        print(img_path)
        img = cv2.imread(img_path)
        img = scaling_img(img)

        base_img_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, base_img_name)
        cv2.imwrite(save_path, img)