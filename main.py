import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,\
                                    GlobalAveragePooling2D, AveragePooling2D, MaxPool2D, UpSampling2D,\
                                    BatchNormalization, Activation, Flatten, Dense, Input,\
                                    Add, Multiply, Concatenate, concatenate, Softmax
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax

tf.keras.backend.set_image_data_format('channels_last')

from skimage.metrics import peak_signal_noise_ratio as psnr

from model import create_model
from data_loader import load_images

model=create_model()

model.load_weights("final_weights.h5")

low_dir = './test/low/'

low_files = [os.path.join(low_dir, file) for file in os.listdir(low_dir)]


train_low_images = load_images(low_files)

train_low_images = np.array(train_low_images)


predicted_images = model.predict(train_low_images)


low_images_dir = './test/low/'

predicted_images_dir = './test/predicted/'
if not os.path.exists(predicted_images_dir):
  os.makedirs(predicted_images_dir)

low_image_filenames = [f for f in os.listdir(low_images_dir) if os.path.isfile(os.path.join(low_images_dir, f))]


for i, (image, filename) in enumerate(zip(predicted_images, low_image_filenames)):
 
  image = np.clip(image, 0, 1) 

  # Handle grayscale images
  if len(image.shape) == 2:
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)

  # Convert to uint8 for saving PNGs
  image = (image * 255).astype('uint8')  # Scale to 0-255 and convert to uint8

  # Save image using Pillow with corresponding filename
  image_pil = Image.fromarray(image)  # Create PIL Image object
  # Extract base filename (without extension) and add "_predicted" suffix
  base_filename = os.path.splitext(filename)[0]
  image_path = os.path.join(predicted_images_dir, f"{base_filename}_predicted.png")
  image_pil.save(image_path)  # Save using PIL