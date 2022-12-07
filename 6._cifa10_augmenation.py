# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:58:12 2021
@author: hasungjae

after CIFA10 using augmentation

"""

import numpy as np
import os
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Flatten,Dense,Dropout,BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
import tkinter as tk
from tkinter import filedialog
import pathlib
from tensorflow import keras


# 1. data ?占쏙옙?占쏙옙/ cifa10 reading ================================================
(x_train, y_train),(x_test, y_test) = cifar10.load_data()


#x_train shape= (50000, 32, 32, 3) ,y_train label shape= (50000, 1)
#x_test shape= (10000, 32, 32, 3) y_test label shape= (10000, 1)
print('x_test shape=',x_test.shape,'y_test label shape=',y_test.shape)
print('\n x_train shape=',x_train.shape,',y_train label shape=',y_train.shape)
print('\n train label=',y_train)# ?占쏙옙?占쏙옙?占쏙옙?占쏙옙?占쏙옙 ?占쏙옙?占쏙옙 異쒕젰
print('test label =', y_test)

# 25占�? ?占쏙옙誘몌옙?? 異쒕젰==============================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6)) 

for index in range(25):    # 25 占�? ?占쏙옙誘몌옙?? 異쒕젰

    plt.subplot(5, 5, index + 1)  # 5?占쏙옙 5?占쏙옙 
    plt.imshow(x_train[index], cmap='gray') 
    plt.axis('off')   

plt.show()

# 1占�? ?占쏙옙誘몌옙?? 異쒕젰===============================================================
plt.figure(figsize=(1/2, 1/2)) 
plt.imshow(x_train[9], cmap='gray') 
plt.colorbar()   
plt.show()

# 2. data ?占쏙옙泥섎━===============================================================

x_train=x_train.reshape(-1, 32, 32, 3)# (?占쏙옙占�? 占�??占쏙옙, 占�?占�?,?占쏙옙占�?, RGB) ?占쏙옙?占쏙옙?占쏙옙占�? 占�??占쏙옙
x_test=x_test.reshape(-1, 32, 32, 3)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train=x_train.astype(np.float32)/255 # normalization(0~1)
x_test=x_test.astype(np.float32)/255

# 3. data augmentation=========================================================

augment_ratio=1.5 # data 150% 利앷컯
augment_size=int(augment_ratio*x_train.shape[0]) #1.5*50000=75000

# ?占쏙옙占�? x_train 媛쒖닔?占쏙옙 150% 
randidx = np.random.randint(x_train.shape[0], size=augment_size)#50000,75000
print('randix:',randidx.shape)

# ?占쏙옙?占쏙옙占�? ?占쏙옙?占쏙옙?占쏙옙 ?占쏙옙?占쏙옙?占쏙옙?占쏙옙 ?占쏙옙蹂몃뜲?占쏙옙?占쏙옙占�? 李몄“?占쏙옙占�? ?占쏙옙臾몄뿉
# ?占쏙옙蹂몃뜲?占쏙옙?占쏙옙?占쏙옙 ?占쏙옙?占쏙옙?占쏙옙 以꾩닔 ?占쏙옙?占쏙옙. 洹몃옒?占쏙옙 copy() ?占쏙옙?占쏙옙占�? ?占쏙옙?占쏙옙 ?占쏙옙?占쏙옙?占쏙옙占�? 蹂듭궗占�? 留뚮벉
x_augmented = x_train[randidx].copy()  
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape)

# 4. ?占쏙옙誘몌옙?? 蹂닿컯 ?占쏙옙?占쏙옙===========================================================
gen = ImageDataGenerator(rotation_range=20,shear_range=0.2,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)

x_augmented, y_augmented = gen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle=False).next()

print(x_augmented.shape, y_augmented.shape)

# 利앷컯?占쏙옙?占쏙옙?占쏙옙 ?占쏙옙?占쏙옙
plt.figure(figsize=(6, 6)) 

for index in range(25):    # 25 占�? ?占쏙옙誘몌옙?? 異쒕젰

    plt.subplot(5, 5, index + 1)  # 5?占쏙옙 5?占쏙옙 
    plt.imshow(x_augmented[index], cmap='gray') 
    plt.axis('off')   

plt.show()

# 利앷컯?占쏙옙?占쏙옙?占쏙옙(75000)+湲곗〈(50000)=125000
# x_train, y_train ?占쏙옙 蹂닿컯?占쏙옙 ?占쏙옙?占쏙옙?占쏙옙 異뷂옙??

x_train_aug= np.concatenate( (x_train, x_augmented) )
y_train_aug = np.concatenate( (y_train, y_augmented) )


# 5. CNN 紐⑤뜽占�?================================================================
model=Sequential()

# Feature map part============
model.add(Conv2D(32,(3,3),activation='relu', padding='same',input_shape=(32,32,3)))
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(32,32,3),activation='relu',padding='same')) 
model.add(Conv2D(64,(3,3),activation='relu',padding='same')) 
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25)) 
model.add(Conv2D(128,(3,3),activation='relu',padding='same')) 
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25)) 
# model.add(Conv2D(256,(3,3),activation='relu',padding='same')) 
# model.add(MaxPooling2D((2,2))) 
# model.add(Dropout(0.25))


model.add(Flatten())

#classfication part=========

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5)),
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

# 6. 紐⑤뜽 而댄뙆?占쏙옙=====================================================================================
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. ?占쏙옙?占쏙옙 ?占쏙옙?占쏙옙(selection 7.1: no augmentation/7.2: augmentation)=====================================

# 7-1 no agumentation (50000)
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=77)

# hist = cnn.fit(x_train, y_train, batch_size=256, epochs=30, validation_data=(x_test, y_test))

# 7-2 augmentation(12500)

hist = model.fit(x_train_aug, y_train_aug, batch_size=656,epochs=300, validation_data=(x_test, y_test),callbacks=[early_stopping_callback])

# 7-3. save model
model.save("Augment_model.h5py")
#cnn.save("non_aungment_modle.h5")
print("Saved model to disk")


# 8. ?占쏙옙?占쏙옙 寃곌낵 ?占쏙옙?占쏙옙?占쏙옙===========================================================
model.evaluate(x_test, y_test)
 

# 9. ?占쏙옙?占쏙옙寃곌낵 占�? ?占쏙옙?占쏙옙?占쏙옙 ?占쏙옙?占쏙옙 洹몃옒?占쏙옙===============================================
import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy Trend')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='best')
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss Trend')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='best')
plt.grid()
plt.show()

