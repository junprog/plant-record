import numpy as np

from PIL import Image
import tensorflow as tf

import matplotlib.pyplot as plt

def read_image(data_path):
    image = tf.io.read_file(data_path)
    image = tf.image.decode_image(image)
    return image

def read_mask(data_path):
    image = tf.io.read_file(data_path)
    image = tf.image.decode_png(image)
    return image

def normalize_image(image):
    image = tf.cast(image, tf.float32) / 255.0
    return image

@tf.function
def _train_parse_function(image_path, mask_path):
    input_image = read_image(image_path)
    input_image = normalize_image(input_image)

    input_mask = read_mask(mask_path)

    #input_image = tf.image.random_crop(input_image, (224, 224, 3), seed=765)
    #input_mask = tf.image.random_crop(input_mask, (224, 224, 1), seed=765)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask

def _test_parse_function(image_path, mask_path):
    input_image = read_image(image_path)
    input_image = normalize_image(input_image)

    input_mask = read_mask(mask_path)

    return input_image, input_mask

def create_dataset(all_image_paths, all_mask_paths, phase='train'):
    tf.random.set_seed(765)
    path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_mask_paths))
    if phase == 'train':
        image_ds = path_ds.map(_train_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif phase == 'val' or phase == 'test':
        image_ds = path_ds.map(_test_parse_function)
    
    return image_ds

if __name__ == '__main__':
    with open("segmentation/train.txt", "r") as tr:
        tr_list = tr.read().split('\n')

    with open("segmentation/val.txt", "r") as vl:
        vl_list = vl.read().split('\n')

    tr_mask_list = [s.replace('.jpg', '_mask.png') for s in tr_list]
    vl_mask_list = [s.replace('.jpg', '_mask.png') for s in vl_list]
    
    tr_ds = create_dataset(tr_list, tr_mask_list, phase='train')
    vl_ds = create_dataset(vl_list, vl_mask_list, phase='val')

    plt.figure(figsize=(16,16))

    N = 16
    for n,(image, mask) in enumerate(tr_ds.take(N)):
        plt.subplot(2,N,2*n+1)
        plt.imshow(image)
        plt.subplot(2,N,2*n+2)
        plt.imshow(np.squeeze(np.array(mask)))
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.show()
