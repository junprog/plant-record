import os
from glob import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

## IMG_SIZE のスコープ問題
def create_dataset(data_path, img_size, batch_size, seed=30):

    training_data = os.path.join(data_path, 'train')
    test_data = os.path.join(data_path, 'test')
    
    TRAINSET_SIZE = len(glob(os.path.join(training_data, "*.jpg")))
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")
    TESTSET_SIZE = len(glob(os.path.join(test_data, "*.jpg")))
    print(f"The Test Dataset contains {TESTSET_SIZE} images.")

    # TODO:configファイル
    SEED = seed
    IMG_SIZE = img_size
    BATCH_SIZE = batch_size

    def parse_image(img_path: str) -> dict:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.uint8)

        '''
        regrex: 正規表現
        Replace elements of input matching regex pattern with rewrite
        tf.strings.regex_replace(input, pattern, rewrite, replace_global=True, name=None)
        '''

        mask_path = tf.strings.regex_replace(img_path, "jpg", "png")
        # print("mask_path {}".format(mask_path))
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
        mask = tf.where(mask == 38, np.dtype('uint8').type(1), mask)
        mask = tf.where(mask == 75, np.dtype('uint8').type(2), mask)
        mask = tf.where(mask == 113, np.dtype('uint8').type(3), mask)

        return {'image': img, 'segmentation_mask': mask}


    @tf.function
    def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:

        input_image = tf.cast(input_image, tf.float32) / 255.0 
        return input_image, input_mask

    @tf.function
    def load_image_train(datapoint: dict) -> tuple:

        input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask

    @tf.function
    def load_image_test(datapoint: dict) -> tuple:

        input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # print(f"Tensorflow ver. {tf.__version__}")

    train_dataset = tf.data.Dataset.list_files(os.path.join(training_data, "*.jpg"), seed=SEED)
    train_dataset = train_dataset.map(parse_image)
    test_dataset = tf.data.Dataset.list_files(os.path.join(test_data, "*.jpg"), seed=SEED)
    test_dataset =test_dataset.map(parse_image)

    dataset = {"train": train_dataset, "test": test_dataset}

    # シャッフルバッファのサイズをデータセットとおなじに設定することで、データが完全にシャッフルされる
    BUFFER_SIZE = TRAINSET_SIZE

    dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    dataset['test'] = dataset['test'].map(load_image_test)
    dataset['test'] = dataset['test'].repeat()
    dataset['test'] = dataset['test'].batch(1)
    dataset['test'] = dataset['test'].prefetch(buffer_size=AUTOTUNE)

    return dataset