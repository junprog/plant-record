import sys
import os
current_path = os.getcwd()
sys.path.append(current_path) # /plant-record/ ディレクトリをパスに追加

import segmentation as seg

if __name__ == '__main__':
    #test
    img_path = 'C:/Users/Junya/Downloads/89hjqj6arfb/20210521_184321402_iOS.jpg'
    weight_path = 'weights/0615-211605-unet'

    mask_path, viz_path = seg.predict(img_path, weight_path)
    print(mask_path, viz_path)