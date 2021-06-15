import sys
import os
current_path = os.getcwd()
sys.path.append(current_path) # /plant-record/ ディレクトリをパスに追加

import segmentation as seg

if __name__ == '__main__':
    #test
    img_path = 'C:\\Users\\jyuny\\Desktop\\addition\\pic_20210424_033612274_iOS.jpg'
    weights_path = 'C:\\Users\\jyuny\\Downloads\\plant_segmentation_data\\0614-160014-unet'

    out_path = seg.predict(img_path, weights_path)
    print(out_path)