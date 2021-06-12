import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

current_path = os.getcwd()
sys.path.append(current_path) # /plant-record/ ディレクトリをパスに追加
from segmentation.models.unet import unet
from tensorflow.keras.optimizers import Adam


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

if __name__ == '__main__':

    IMG_SIZE = 128
    N_CHANNELS = 3
    N_CLASSES = 4

    input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)
    num_classes = N_CLASSES

    #test
    img_path = 'PATH_TO_IMAGE'
    weights_path = 'PATH_TO_YOUR_WEIGHT'

    #input
    img = tf.io.read_file(img_path)
    img_tensor = preprocess_image(img)
    img_tensor_reshape = tf.expand_dims(img_tensor, axis=0)
    
    # model = unet(input_size, num_classes=num_classes)
    model = tf.keras.models.load_model(weights_path)
    # model.compile(optimizer=Adam(learning_rate=0.0005), loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    #             metrics=['accuracy'])
    pred_mask = model.predict(img_tensor_reshape)

    display_sample([img_tensor, create_mask(pred_mask)])