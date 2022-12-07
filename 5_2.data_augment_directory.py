import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# Using by ImageDataGenerator ..................

gen = ImageDataGenerator(rotation_range = 30,
                        width_shift_range = 0.3,
                        height_shift_range=0.2,
                        shear_range=0.3,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='reflect',
                        )                      #Also try nearest, constant, reflect, wrap

data_path = 'C:/Users/user2/GH/Deep_learning/data_aug/aug_image' # To augment image directory here 3classes and each 4file

i=0
for batch in gen.flow_from_directory(directory=data_path,
                                batch_size=2,
                                shuffle=True,
                                target_size=(100, 100),
                                color_mode="rgb",
                                save_to_dir='C:/Users/user2/GH/Deep_learning/data_aug/aug_data_directory',
                                save_prefix='aug',
                                save_format='png',
                                class_mode='binary'
                                ):
    i+=1
    if i> 31:
        break


