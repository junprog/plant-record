import sys
import os
import argparse
current_path = os.getcwd()
sys.path.append(current_path) # /plant-record/ ディレクトリをパスに追加

import segmentation as seg

def parse_args():
    parser = argparse.ArgumentParser(description='Inference Segmentation Model')
    parser.add_argument('--image-path', default='C:/Users/Junya/Downloads/89hjqj6arfb/20210521_184321402_iOS.jpg',
                        help='input image path')
    parser.add_argument('--weight-path', default='weights/0615-225649-unet',
                        help='weight path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    #test
    args = parse_args()
    img_path = args.image_path
    weight_path = args.weight_path

    mask_path, viz_path = seg.predict(img_path, weight_path)
    print(mask_path, viz_path)