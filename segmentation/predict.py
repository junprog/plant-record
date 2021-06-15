import os
import sys
import datetime

import cv2
import imgviz
import labelme
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
current_path = os.getcwd()
sys.path.append(current_path) # /plant-record/ ディレクトリをパスに追加

from segmentation.models.unet import unet

IMG_SIZE = 128
N_CHANNELS = 3
N_CLASSES = 4

def preprocess_image(image):

	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.convert_image_dtype(image, tf.uint8)
	image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	image = tf.cast(image, tf.float32) / 255.0

	return image

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:

    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.squeeze(pred_mask, axis=0)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)

    return pred_mask
    
def display_sample(display_list):
  
	plt.figure(figsize=(10, 10))
	title = ['Input Image', 'Predicted Mask']

	for i in range(len(display_list)):
		plt.subplot(1, len(display_list), i+1)
		plt.title(title[i])
		plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
		plt.axis('off')
	plt.show()

def mask_tf2np(mask):

	mask = create_mask(mask)
	mask = tf.keras.preprocessing.image.array_to_img(mask)
	mask = np.array(mask)
	
	return mask

def create_vizmask(image, mask, output_img_path):

	idx_255 = np.where(mask==255)
	mask[idx_255] = 254
	labelme.utils.lblsave(output_img_path, mask)

"""

# or kerassurgeon for standalone Keras
from tfkerassurgeon import delete_layer, insert_layer

model = delete_layer(model.layers[0])
# inserts before layer 0
model = insert_layer(model.layers[0], new_input)
"""

def predict(input_img_path, weight_path):
	"""
	Args: 
		input_img_path    --- 入力画像のpath
		weight_path       --- 学習済みモデルのpath
	Return:
		output_img_path   --- 出力されたmask画像のpath

	"""

	input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)
	num_classes = N_CLASSES

	# input
	img = tf.io.read_file(input_img_path)
	img_tensor = preprocess_image(img)
	w, h, _ = img_tensor.shape
	img_tensor_reshape = tf.expand_dims(img_tensor, axis=0)
	np_img = np.array(tf.keras.preprocessing.image.array_to_img(img_tensor))

	# model load & prediction
	model = tf.keras.models.load_model(weight_path)
	new_input = tf.keras.layers.Input(shape=(w, h, 3), name='image_input')

	pred_mask = model.predict(img_tensor_reshape)

	# save mask 
	# save_path: output_img_path
	input_img_name = os.path.basename(input_img_path)
	output_img_name = input_img_name.replace('pic_', 'mask_')
	output_img_name = output_img_name.replace('.jpg', '.png')

	current_path = os.getcwd()
	img_dir = os.path.join(current_path, 'images')
	if os.path.isdir(img_dir) == False:
		os.mkdir(img_dir)
	
	output_img_path = os.path.join(img_dir, output_img_name)
	pred_mask = mask_tf2np(pred_mask)

	create_vizmask(np_img, pred_mask, output_img_path)

	return output_img_path

if __name__ == '__main__':

    #test
    img_path = 'C:\\Users\\jyuny\\Desktop\\addition\\20210423_025925190_iOS.jpg'
    weights_path = 'C:\\Users\\jyuny\\Downloads\\plant_segmentation_data\\0614-152620-unet'

    predict(img_path, weight_path)
    
    display_sample([img_tensor, create_mask(pred_mask)])