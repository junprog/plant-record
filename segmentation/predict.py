import os
import sys

import labelme
import imgviz

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
current_path = os.getcwd()
sys.path.append(current_path) # /plant-record/ ディレクトリをパスに追加

from segmentation.models.unet import unet

MIN_SIZE = 384

def search_input_shape(image):
	"""
	UNet の入力サイズの探索を行う。
	e.g.) 	1st Step:
				2031 x 4067 -> MIN_SIZE x MIN_SIZE*round(4067 / 2031) = 512 x 1024
		  	2nd Step:
			  	512 x 1024 -> 32*round(512 / 32) x 32*round(1024 / 32)  32の倍数で近い値に変更 (32という数字はUnetのconcatする際にシェイプの不一致をなくすため)

	Args:
		image	--- 入力画像 [tensorflow.python.framework.ops.EagerTensor]
	Return:
		new_w	--- UNetの入力幅
		new_h	--- UNetの入力高さ
	"""

	w, h, _ = image.shape
	if w > h:
		scale = w / h
		new_h = MIN_SIZE
		new_w = new_h * scale
	else:
		scale = h / w
		new_w = MIN_SIZE
		new_h = new_w * scale

	new_w = round(new_w / 32) * 32
	new_h = round(new_h / 32) * 32

	return new_w, new_h

def preprocess_image(image):
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.convert_image_dtype(image, tf.uint8)
	new_shape = search_input_shape(image)
	image = tf.image.resize(image, new_shape)
	image = tf.cast(image, tf.float32) / 255.0

	return image

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
	pred_mask = tf.argmax(pred_mask, axis=-1)
	pred_mask = tf.squeeze(pred_mask, axis=0)
	# pred_mask becomes [IMG_SIZE, IMG_SIZE]
	# but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
	pred_mask = tf.expand_dims(pred_mask, axis=-1)

	return pred_mask

def mask_tf2np(mask):
	mask = create_mask(mask)
	mask = tf.keras.preprocessing.image.array_to_img(mask)
	mask = np.array(mask)

	if 255 in np.unique(mask):
		idx = np.where(mask==255)
		mask[idx] = 254 
	
	return mask

def create_classnames(label_path='dataset/labels.txt'):
	"""
	imgviz.io.imsave で必要な可視化画像を作成するためクラス名を取得し tuple で返す

	Args:
		label_path	--- dataset/generate_mask.py で使用した labels.txt の保存ファイル名
	Return:
		class_names	--- クラス名が格納された tuple
	"""

	class_names = []
	class_name_to_id = {}
	for i, line in enumerate(open(label_path).readlines()):
		class_id = i - 1  # starts with -1
		class_name = line.strip()
		class_name_to_id[class_name] = class_id
		if class_id == -1:
			assert class_name == "__ignore__"
			continue
		elif class_id == 0:
			assert class_name == "_background_"
		class_names.append(class_name)
	class_names = tuple(class_names)
	
	return class_names

def create_save_vizmask(image, mask, output_mask_path, output_viz_path):
	"""
	カラーリングされたセグメンテーションマスク: {}_mask.png の作成, 保存
	入力画像とセグメンテーションマスクをブレンダリングした可視化画像: {}_viz.png の作成, 保存

	Args: 
		image    			--- 入力画像
		mask    			--- セグメンテーションマスク
		output_mask_path	--- セグメンテーションマスクの保存ファイル名
		output_viz_path		--- 可視化画像の保存ファイル名
	"""

	# 入力画像
	image = np.array(tf.keras.preprocessing.image.array_to_img(image))

	# 予測した mask を labelme.utils.lblsave でカラーリングできる形 mask に変換
	mask = mask_tf2np(mask)

	# imgviz.io.imsave で可視化画像を作成するためクラス名を取得
	class_names = create_classnames()

	# 予測した mask を imgviz.io.imsave で可視化画像を作成できる形 viz に変換
	viz_mask = np.zeros_like(mask)
	mask_set = np.unique(mask)
	for i, s in enumerate(mask_set):
		idx = np.where(mask==s)
		viz_mask[idx] = i
	viz = imgviz.label2rgb(label=viz_mask, img=imgviz.rgb2gray(image), font_size=15, label_names=class_names, loc="rb")

	labelme.utils.lblsave(output_mask_path, mask)
	imgviz.io.imsave(output_viz_path, viz)


def predict(input_img_path, weight_path):
	"""
	Args: 
		input_img_path    	--- 入力画像のpath
		weight_path       	--- 学習済みモデルのpath
	Return:
		output_mask_path   	--- 出力されたセグメンテーションマスクの保存ファイル名
		output_viz_path		--- 可視化画像の保存ファイル名
	"""

	# input
	img = tf.io.read_file(input_img_path)
	img_tensor = preprocess_image(img)
	
	input_size = img_tensor.shape
	num_classes = 4

	img_tensor_reshape = tf.expand_dims(img_tensor, axis=0)

	# model load & prediction
	alt_shape_model = unet(input_size, num_classes=num_classes)
	model = tf.keras.models.load_model(weight_path)

	for new_layer, layer in zip(alt_shape_model.layers[1:], model.layers[1:]):
		new_layer.set_weights(layer.get_weights())

	pred_mask = alt_shape_model.predict(img_tensor_reshape)

	# save mask 
	# save_path: output_img_path
	input_img_name = os.path.basename(input_img_path)
	output_mask_name = input_img_name.replace('.jpg', '_mask.png')
	output_viz_name = input_img_name.replace('.jpg', '_viz.png')

	current_path = os.getcwd()
	img_dir = os.path.join(current_path, 'images')
	if os.path.isdir(img_dir) == False:
		os.mkdir(img_dir)
	
	output_mask_path = os.path.join(img_dir, output_mask_name)
	output_viz_path = os.path.join(img_dir, output_viz_name)

	create_save_vizmask(img_tensor, pred_mask, output_mask_path, output_viz_path)

	return output_mask_path, output_viz_path


if __name__ == '__main__':

	#test
	img_path = 'C:/Users/Junya/Downloads/89hjqj6arfb/20210521_184335231_iOS.jpg'
	weight_path = 'D:/Junya/Documents/plant-record-res/0615-202111-unet'

	predict(img_path, weight_path)