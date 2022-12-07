# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 23:04:06 2021

@author: hasungjae

using flow single image augmentation

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# img = load_img('C:/Users/hasun/OneDrive/바탕 ?���?/?��기술교육?��/preview')

# this is a PIL image
img = load_img('C:/Users/user2/GH/Deep_learning/data_aug/aug_image/deer/deer1.jpg')
x1 = img_to_array(img)  # this is a Numpy array with shape (183,275,3)
# this is a Numpy array with shape (1,183,275,3)
x = x1.reshape((1,) + x1.shape)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `aug_data/` folder
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='aug_data', save_prefix='dog', save_format='jpeg'):
    i += 1
    if i > 1000:
        break  # otherwise the generator would loop indefinitely
