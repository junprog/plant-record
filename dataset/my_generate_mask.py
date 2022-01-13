"""
labelme で作成したjsonファイルをpngにして*_mask.pngというファイル名で保存
"""
import os
import glob
import json
import argparse
import numpy as np
from PIL import Image
import labelme.utils as utils
from matplotlib import pyplot as plt

# 0は背景クラス
CLASSES = {
    'leaf': 1,
    'stem': 2,
    'pot': 3,
}

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Mask Args')
    parser.add_argument('--data-dir', default='C:/Users/junya/Documents/plant_segmentation_data',
                        help='dataset directory')
    args = parser.parse_args()
    return args

def load_json(json_path):
    """
    Args:
        json_path: jsonファイルのパス
    Return:
        jsonを読み込んだインスタンス
    """
    with open(json_path, "r",encoding="utf-8") as f:
        return json.load(f)

def save_np2png(np_array, save_path):
    """
    Args:
        np_array: 2次元numpy配列
        save_path: 保存するpngのパス+ファイル名
    """
    assert not os.path.exists(save_path), "指定したファイル名{}は存在します。".format(save_path)

    print(save_path)
    image = Image.fromarray(np_array)
    image.save(save_path)

def annotation2np(annotation):
    """
        Args:
            annotation: labelmeで作成したjsonを読み込んだ変数
        Return:
            output_mask: 0...クラス数 で構成されたnumpy型のマスク
    """
    height = annotation['imageHeight']
    width = annotation['imageWidth']
    output_mask = np.zeros((height, width), dtype=int)
    bool_masks = dict()
    for label in CLASSES.keys():
        bool_masks[label] = np.zeros((height, width), dtype=bool)
    for shape in annotation['shapes']:
        mask = utils.shape_to_mask((height, width), shape['points'], shape_type=None, line_width=1, point_size=1)
        if shape['label'] in CLASSES:
            bool_masks[shape['label']] += mask
    for label in CLASSES.keys():
        output_mask[bool_masks[label]] = 0
        output_mask += np.where(bool_masks[label]==True, CLASSES[label], 0)
    plt.imshow(output_mask)
    plt.show()
    
    return output_mask

def generate_maskdata():
    args = parse_args()
    for phase in ['train', 'test']:
        data_dir = os.path.join(args.data_dir, phase)
        json_list = glob.glob(os.path.join(data_dir, '*.json'))
        for json_path in json_list:
            annotation = load_json(json_path)
            mask = annotation2np(annotation)
            save_path = json_path.replace('.json', '.png')
            save_np2png(mask, save_path)

if __name__ == '__main__':
    generate_maskdata()