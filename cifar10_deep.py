from keras.datasets import cifar10
from keras.utils import np_utils
from keras. models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D , BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context


seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

(X_train,Y_train),(X_test,Y_test) = cifar10.load_data()
X_train = X_train.reshape(-1, 32, 32, 3).astype(np.float32) / 255
X_test = X_test.reshape(-1, 32, 32, 3).astype(np.float32) / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(32,32,3),activation='relu',padding='same')) 
model.add(BatchNormalization()) 
model.add(Conv2D(64,(3,3),activation='relu',padding='same')) 
model.add(BatchNormalization()) 
model.add(MaxPooling2D((2,2))) 
model.add(Conv2D(128,(3,3),activation='relu',padding='same')) 
model.add(BatchNormalization()) 
model.add(MaxPooling2D((2,2))) 
model.add(Conv2D(256,(3,3),activation='relu',padding='same')) 
model.add(BatchNormalization()) 
model.add(MaxPooling2D((2,2))) 
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.h5py"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10)

history = model.fit(X_train,Y_train, validation_data=(X_test,Y_test) , epochs=30, batch_size= 200, verbose=1,callbacks=[early_stopping_callback,checkpointer])

print("\nloss : %.4f\nAccuracy : %.4f" %(model.evaluate(X_test,Y_test)[0],model.evaluate(X_test,Y_test)[1]))

y_vacc=history.history['val_accuracy']
y_acc=history.history['accuracy']

x_len=np.arange(len(y_acc))
plt.plot(x_len,y_vacc,marker=".",c="red",markersize=3,label='Testset_acc')
plt.plot(x_len,y_acc,marker=".",c="blue",markersize=3,label='Trainset_acc')

plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()