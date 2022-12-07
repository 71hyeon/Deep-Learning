from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import tensorflow as tf

np.random.seed(3)           # def
tf.random.set_seed(3)       #
Data_set = np.loadtxt("C:/Users/user2/Desktop/Gihyeon/dataset/pima-indians-diabetes.csv",delimiter = ",")

X = Data_set[:,0:8]
Y = Data_set[:,8]

model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(X,Y, epochs = 200, batch_size = 10)

print("\n loss: %.4f \n Accuracy: %.4f"%(model.evaluate(X,Y)[0],model.evaluate(X,Y)[1]))
#print("\n Accuracy: %.4f"%(model.evaluate(X,Y)[1]))