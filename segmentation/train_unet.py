import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import argparse
from datetime import datetime
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

current_path = os.getcwd()
sys.path.append(current_path) # /plant-record/ ディレクトリをパスに追加
from segmentation.models.unet import unet
from segmentation.load_dataset import create_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Training Segmentation Model')
    parser.add_argument('--data-dir', default='C:/Users/Yoshimoto/Downloads/plant_segmentation_data',
                        help='dataset directory')
    parser.add_argument('--result-dir', default='C:/Users/Yoshimoto/Documents/plant-record-res',
                        help='result directory')

    parser.add_argument('--gpu', action='store_true',
                    help='use GPU')

    parser.add_argument('--model-arch', default='unet',
                    help='model architecture [unet, ]')

    args = parser.parse_args()
    return args

"""
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

    return {'image': img, 'segmentation_mask': mask}


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:

    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:

    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:

    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask
"""

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:

    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask

def show_predictions(dataset=None, num=1):

    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display_sample([image[0], mask[0], create_mask(pred_mask)])
    else:
        # The model is expecting a tensor of the size
        # [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
        # but sample_image[0] is [IMG_SIZE, IMG_SIZE, 3]
        # and we want only 1 inference to be faster
        # so we add an additional dimension [1, IMG_SIZE, IMG_SIZE, 3]
        one_img_batch = sample_image[0][tf.newaxis, ...]
        # one_img_batch -> [1, IMG_SIZE, IMG_SIZE, 3]
        inference = model.predict(one_img_batch)
        # inference -> [1, IMG_SIZE, IMG_SIZE, N_CLASS]
        pred_mask = create_mask(inference)
        # pred_mask -> [1, IMG_SIZE, IMG_SIZE, 1]
        display_sample([sample_image[0], sample_mask[0],
                        pred_mask[0
                        ]])


def display_sample(display_list):

    plt.figure(figsize=(10, 10))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':

    args = parse_args()

    SEED = 30
    IMG_SIZE = 448
    N_CHANNELS = 3
    BATCH_SIZE = 1
    N_CLASSES = 4

    training_data = os.path.join(args.data_dir, 'train')
    test_data = os.path.join(args.data_dir, 'test')
    
    TRAINSET_SIZE = len(glob(os.path.join(training_data, "*.jpg")))
    TESTSET_SIZE = len(glob(os.path.join(test_data, "*.jpg")))

    dataset = create_dataset(args.data_dir, IMG_SIZE, BATCH_SIZE)

    for image, mask in dataset['train'].take(1):
        sample_image, sample_mask = image, mask

#    display_sample([sample_image[0], sample_mask[0]])

    # -- Keras Functional API -- #
    # -- UNet Implementation -- #
    dropout_rate = 0.5
    input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)
    num_classes = N_CLASSES

    model = unet(input_size, num_classes=num_classes)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    model.summary()

    model.summary()

    for image, mask in dataset['train'].take(1):
        sample_image, sample_mask = image, mask

#    show_predictions()
    EPOCHS = 1


    STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
    VALIDATION_STEPS = TESTSET_SIZE // BATCH_SIZE

    checkpoint_name = "cp.ckpt"
    
    ### 結果出力ディレクトリ作成 ###
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    save_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = save_dir + '-{}'.format(args.model_arch)
    save_dir = os.path.join(args.result_dir, save_dir)
    os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, checkpoint_name)

    # チェックポイントコールバックを作る
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                    save_weights_only=True,
                                                    verbose=1)
    
    if args.gpu:
        # On GPU
        model_history = model.fit(dataset['train'], epochs=EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_steps=VALIDATION_STEPS,
                                validation_data=dataset['test'],
                                callbacks=[cp_callback])

    else:
        # On CPU
        with tf.device("/cpu:0"):
            print("CPU\n\n\n")
            model_history = model.fit(dataset['train'], epochs=EPOCHS,
                                    steps_per_epoch=STEPS_PER_EPOCH,
                                    validation_steps=VALIDATION_STEPS,
<<<<<<< HEAD
                                    validation_data=dataset['test'])#,
                                    #callbacks=[cp_callback])
    
=======
                                    validation_data=dataset['test'],
                                    callbacks=[cp_callback])

    show_predictions()
>>>>>>> ca2aff431bcfe322a2a5033b7620da1c5d39a251
