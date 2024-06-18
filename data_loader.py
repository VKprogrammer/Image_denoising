# from google.colab import drive
# drive.mount('/content/drive')

import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



# base_dir = '/content/drive/My Drive/Train'
# low_dir = os.path.join(base_dir, 'low')
# high_dir = os.path.join(base_dir, 'high')


# low_files = [os.path.join(low_dir, file) for file in os.listdir(low_dir)]
# high_files = [os.path.join(high_dir, file) for file in os.listdir(high_dir)]


# paired_files = []
# for low_file in low_files:
#     low_filename = os.path.basename(low_file)
#     matching_high_file = os.path.join(high_dir, low_filename)
#     if os.path.exists(matching_high_file):
#         paired_files.append((low_file, matching_high_file))


# total_samples = len(paired_files)
# print(total_samples)
# train_ratio = 0.6
# val_ratio = 0.2
# test_ratio = 0.2



import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
# train_low_files, valtest_low_files = train_test_split(low_files, test_size=(val_ratio + test_ratio), random_state=42)
# val_low_files, test_low_files = train_test_split(valtest_low_files, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)

# train_high_files = [os.path.join(high_dir, os.path.basename(file)) for file in train_low_files]
# val_high_files = [os.path.join(high_dir, os.path.basename(file)) for file in val_low_files]
# test_high_files = [os.path.join(high_dir, os.path.basename(file)) for file in test_low_files]

def load_and_preprocess_image(file_path):
    image = Image.open(file_path).convert('RGB')  # Ensure the image is in RGB format
    image = image.resize((256, 256))  # Resize to a fixed size if needed
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image
def load_images(image_files):
    filtered_files = [file for file in image_files if not os.path.basename(file).startswith('.')]
    return [load_and_preprocess_image(file) for file in filtered_files]
# def load_images(image_files):
#     return [load_and_preprocess_image(file) for file in image_files]

# Load and preprocess the data
# train_low_images = load_images(train_low_files)
# train_high_images = load_images(train_high_files)
# val_low_images = load_images(val_low_files)
# val_high_images = load_images(val_high_files)
# test_low_images = load_images(test_low_files)
# test_high_images = load_images(test_high_files)

# def display_pairs(low_images, high_images, title):
#     plt.figure(figsize=(10, 4))
#     for i in range(2):
#         plt.subplot(2, 2, i*2 + 1)
#         low_img = cv2.imread(low_images[i])
#         low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
#         plt.imshow(low_img)
#         plt.title('Low Quality')
#         plt.axis('off')

#         plt.subplot(2, 2, i*2 + 2)
#         high_img = cv2.imread(high_images[i])
#         high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
#         plt.imshow(high_img)
#         plt.title('High Quality')
#         plt.axis('off')
#     plt.suptitle(title)
#     plt.show()

# # Display images for training set
# display_pairs(train_low_images[:10], train_high_images[:10], title='Training Set')

# # Display images for validation set
# display_pairs(val_low_images[:10], val_high_images[:10], title='Validation Set')

# # Display images for testing set
# display_pairs(test_low_images[:10], test_high_images[:10], title='Testing Set')


# Convert the images to numpy arrays
# train_low_images = np.array(train_low_images)
# train_high_images = np.array(train_high_images)
# val_low_images = np.array(val_low_images)
# val_high_images = np.array(val_high_images)
