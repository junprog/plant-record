import os
import glob
import argparse

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Display datas')
    parser.add_argument('--data-dir', default='C:/Users/junya/Documents/plant_segmentation_data',
                        help='dataset directory')
    parser.add_argument('--val-rate', default=0.1, type=float,
                        help='Number of validation rate')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    phase = 'train'
    train_dir = os.path.join(args.data_dir, phase)
    
    all_set = np.array(glob.glob(os.path.join(train_dir, '*.jpg')))
    all_idx = np.random.choice(all_set.shape[0], all_set.shape[0], replace=False)

    train_num = int(all_set.shape[0] * (1 - args.val_rate))
    train_idx = all_idx[0:train_num]
    val_idx = all_idx[train_num:all_set.shape[0]]

    train_set = all_set[train_idx]
    val_set = all_set[val_idx]

    np.savetxt("segmentation/train.txt", train_set, fmt='%s', delimiter=',')
    np.savetxt("segmentation/val.txt", val_set, fmt='%s', delimiter=',')
