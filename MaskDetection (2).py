# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
import tkinter as tk
from tkinter import filedialog

# Data set ===================================================================

# Data_dir="C:\Python38\DeepLearning\MaskTrain"

data_dir="C:/Users/user2/GH/Deep_learning/Face Mask Dataset/Train"
data_dir=pathlib.Path(data_dir)
batch_size=16
img_height=64
img_width=64

# Reading Training Image from directory ======================================

train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,validation_split=0.2,subset="training",seed=123,
    image_size=(img_height,img_width),batch_size=batch_size)

# Reading Validation images from directory ===================================

val_ds=tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,validation_split=0.2,subset="validation",seed=123,
    image_size=(img_height,img_width),batch_size=batch_size)
    
class_names=train_ds.class_names
print(class_names)

# Memory optimization and speed up excution ==================================

AUTOTUNE=tf.data.experimental.AUTOTUNE 
train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds=val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes=2

# Defining CNN Modeling ======================================================

model=Sequential([
    
    layers.experimental.preprocessing.Rescaling(1./255,input_shape=(img_height,img_width,3)),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),  
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),  
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(num_classes)
                
    ])

# Training the mode ==========================================================

noepochs=7
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
mymodel=model.fit(train_ds,validation_data=val_ds,epochs=noepochs)

acc=mymodel.history['accuracy']
val_acc=mymodel.history['val_accuracy']
loss= mymodel.history['loss']
val_loss=mymodel.history['val_loss']

epochs_range=range(noepochs)
plt.figure(figsize=(15,15))
plt.plot(1,2,1)
plt.plot(epochs_range,loss,label='Training Loss')
plt.plot(epochs_range,val_loss,label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and validation loss')
plt.show()

# Function to Test single image ==============================================

def recogout():
    
    root=tk.Tk()
    root.withdraw()
    
    img_path=filedialog.askopenfilename()
    img=keras.preprocessing.image.load_img(img_path,target_size=(img_height,img_width))
    img_array=keras.preprocessing.image.img_to_array(img)
    img_array=tf.expand_dims(img_array,0)
    predictions=model.predict(img_array)
    score=tf.nn.softmax(predictions[0])
    print("This image most likely belong to {} with a {:.2f} percent conflidence."
    .format(class_names[np.argmax(score)],100*np.max(score)))
    