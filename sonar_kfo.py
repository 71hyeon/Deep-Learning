from ast import Str
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
import tensorflow as tf

a=0

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv('C:/Users/user2/Desktop/Gihyeon/dataset/sonar.csv', header = None)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

accuracy = []
for train, test in skf.split(X,Y):
    model = Sequential()
    model.add(Dense(24, input_dim = 60, activation='relu'))
    model.add(Dense(18, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.fit(X[train],Y[train],epochs=100,batch_size=5)
    k_accuracy = "%.4f"%(model.evaluate(X[test],Y[test])[1])
    accuracy.append(k_accuracy)
    a+=float(k_accuracy)




print("\n%.f foldaccuracy\n" %n_fold,accuracy)
print("average = %.4f" %(a/n_fold))