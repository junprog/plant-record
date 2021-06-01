import os
import glob
import argparse
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='D:/Junya/Documents/plant_segmentation_data',
                        help='dataset directory')
    parser.add_argument('--result-dir', default='D:/Junya/Documents/plant-record-res',
                        help='result directory')
    args = parser.parse_args()
    return args

def read_image(data_path):
    image = tf.io.read_file(data_path)
    img_tensor = tf.image.decode_image(image)

    return img_tensor

def transform_image(image):
    out = tf.image.resize_images(image, )

if __name__ == '__main__':
    args = parse_args()

    for phase in ['train', 'test']:
        data_dir = os.path.join(args.data_dir, phase)
        image_list = glob.glob(os.path.join(data_dir, '*.jpg'))
        print(image_list)

        for img_path in image_list:
            img_tensor = read_image(img_path)
            print(img_tensor)