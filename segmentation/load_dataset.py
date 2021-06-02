import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from glob import glob

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

def display_sample(display_list):

    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


if __name__ == '__main__':

    training_data = 'train/'
    test_data = 'test/'
    
    TRAINSET_SIZE = len(glob(training_data + "*.jpg"))
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")
    TESTSET_SIZE = len(glob(test_data + "*.jpg"))
    print(f"The Test Dataset contains {TESTSET_SIZE} images.")

    # TODO:configファイル
    SEED = 30
    IMG_SIZE = 128
    BATCH_SIZE = 1

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # print(f"Tensorflow ver. {tf.__version__}")

    train_dataset = tf.data.Dataset.list_files(training_data + "*.jpg", seed=SEED)
    train_dataset = train_dataset.map(parse_image)
    test_dataset = tf.data.Dataset.list_files(test_data + "*.jpg", seed=SEED)
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
    dataset['test'] = dataset['test'].batch(BATCH_SIZE)
    dataset['test'] = dataset['test'].prefetch(buffer_size=AUTOTUNE)

    for image, mask in dataset['train'].take(1):
        sample_image, sample_mask = image, mask

    display_sample([sample_image[0], sample_mask[0]])