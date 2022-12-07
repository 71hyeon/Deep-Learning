from pytest import importorskip
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from iris_multi_pd import Y_encoded

# seedê°? ?„¤? •
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# Data ?…? ¥
df = pd.read_csv('C:/Users/user2/Desktop/Gihyeon/dataset/iris.csv',names = ["sepal_length","sepal_width", "petal_length","petal_width","species"])

# ê·¸ë˜?”„ë¡? ?™•?¸
sns.pairplot(df,hue='species')
plt.show()

# Data ë¶„ë¥˜
dataset = df.values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]

# ë¬¸ì?—´?„ ?ˆ«?ë¡? ë³??™˜ (?› ?•« ?¸ì½”ë”©)
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = np_utils.to_categorical(Y)

# ëª¨ë¸ ?„¤? •
model = Sequential()
model.add(Dense(16,input_dim = 4 , activation='relu'))
model.add(Dense(3, activation='softmax'))

# ëª¨ë¸ ì»´íŒŒ?¼
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# ëª¨ë¸ ?‹¤?–‰
model.fit(X,Y_encoded,epochs=50,batch_size=1)

# ê²°ê³¼ ì¶œë ¥
print("\nloss : %.4f \nAccuracy : %.4f" %(model.evaluate(X,Y_encoded)[0],model.evaluate(X,Y_encoded)[1]))