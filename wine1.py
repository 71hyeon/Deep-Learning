from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import pandas as pd
import numpy
import tensorflow as tf

numpy.random.seed(3)
tf.random.set_seed(3)

df_pre=pd.read_csv('C:/Users/user2/Desktop/Gihyeon/dataset/wine.csv', header=None)
df=df_pre.sample(frac=0.15)

dataset=df.values
X=dataset[:,0:12]
Y=dataset[:,12]

model = Sequential()
model.add(Dense(30,input_dim=12,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

modelpath="./model/{epoch:02d}-{val_loss:.4f}.h5py"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',verbose=1,save_best_only=True)

early_stopping_callback=EarlyStopping(monitor='val_loss',patience=30)


history=model.fit(X,Y,validation_split=0.2, epochs=3500,verbose=0, batch_size=500, callbacks=[early_stopping_callback,checkpointer])

print("\n Accuracy: %.4f" % (model.evaluate(X,Y)[1]))

y_vloss=history.history['val_loss']
y_acc=history.history['accuracy']

x_len=numpy.arange(len(y_acc))
plt.plot(x_len,y_vloss,"o",c="red",markersize=3)
plt.plot(x_len,y_acc,"o",c="blue",markersize=3)

plt.show()