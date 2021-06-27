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
from tensorflow_examples.models.pix2pix import pix2pix
current_path = os.getcwd()
sys.path.append(current_path) # /plant-record/ ディレクトリをパスに追加
from segmentation.models.unet import unet
from segmentation.load_dataset import create_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Training Segmentation Model')
    parser.add_argument('--data-dir', default='',
                        help='dataset directory')
    parser.add_argument('--result-dir', default='weights/',
                        help='result directory')

    parser.add_argument('--gpu', action='store_true',
                    help='use GPU')

    parser.add_argument('--model-arch', default='unet',
                    help='model architecture [unet, ]')

    args = parser.parse_args()
    return args

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

    plt.figure(figsize=(10, 6))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def display_result(loss, val_loss, save_dir):
    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'result.png'))  
    plt.show()


if __name__ == '__main__':

    args = parse_args()

    # TODO:configファイル
    SEED = 30
    IMG_SIZE = 512
    N_CHANNELS = 3
    BATCH_SIZE = 2
    N_CLASSES = 4

    training_data = os.path.join(args.data_dir, 'train')
    test_data = os.path.join(args.data_dir, 'test')
    
    TRAINSET_SIZE = len(glob(os.path.join(training_data, "*.jpg")))
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")
    TESTSET_SIZE = len(glob(os.path.join(test_data, "*.jpg")))
    print(f"The Test Dataset contains {TESTSET_SIZE} images.")

    TRAINSET_SIZE = len(glob(os.path.join(training_data, "*.jpg")))
    TESTSET_SIZE = len(glob(os.path.join(test_data, "*.jpg")))

    dataset = create_dataset(args.data_dir, IMG_SIZE, BATCH_SIZE)

    for image, mask in dataset['test'].take(1):
        sample_image, sample_mask = image, mask

#    display_sample([sample_image[0], sample_mask[0]])

    # -- Keras Functional API -- #
    # -- UNet Implementation -- #
    dropout_rate = 0.5
    input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)
    num_classes = N_CLASSES

    model = unet(input_size, num_classes=num_classes)

    # model = unet_model(OUTPUT_CHANNELS)
    model.compile(optimizer=Adam(learning_rate=0.005),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    EPOCHS = 20

    STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
    VALIDATION_STEPS = TESTSET_SIZE // BATCH_SIZE

    checkpoint_name = "cp.ckpt"
    checkpoint_name = "model_{epoch:06d}.h5"
    checkpoint_name = "saved_model"
    ### 結果出力ディレクトリ作成 ###
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    save_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = save_dir + '-{}'.format(args.model_arch)
    save_dir = os.path.join(args.result_dir, save_dir)
    os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, checkpoint_name)

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
        with tf.device("/CPU:0"):
            model_history = model.fit(dataset['train'], epochs=EPOCHS,
                                    steps_per_epoch=STEPS_PER_EPOCH,
                                    validation_steps=VALIDATION_STEPS,
                                    validation_data=dataset['test'],
                                    callbacks=[cp_callback],
                                    )

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    model.save(save_dir)

    display_result(loss, val_loss, save_dir)
    show_predictions()