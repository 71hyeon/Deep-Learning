from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv('C:/Users/user2/Desktop/Gihyeon/dataset/wine.csv', header = None)

dataset = df.values
X = dataset[:,0:12].astype(float)
Y_obj = dataset[:,12]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=seed)

# model = Sequential()
# model.add(Dense(24, input_dim = 60, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# model.fit(X_train,Y_train,epochs=130,batch_size=5)
# model.save('my_model.h5')

# del model
model = load_model('my_model.h5py')

print("\nTest Accuracy : %.4f\n" %(model.evaluate(X_test,Y_test))[1])