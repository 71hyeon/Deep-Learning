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

# seed�? ?��?��
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# Data ?��?��
df = pd.read_csv('C:/Users/user2/Desktop/Gihyeon/dataset/iris.csv',names = ["sepal_length","sepal_width", "petal_length","petal_width","species"])

# 그래?���? ?��?��
sns.pairplot(df,hue='species')
plt.show()

# Data 분류
dataset = df.values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]

# 문자?��?�� ?��?���? �??�� (?�� ?�� ?��코딩)
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = np_utils.to_categorical(Y)

# 모델 ?��?��
model = Sequential()
model.add(Dense(16,input_dim = 4 , activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파?��
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 모델 ?��?��
model.fit(X,Y_encoded,epochs=50,batch_size=1)

# 결과 출력
print("\nloss : %.4f \nAccuracy : %.4f" %(model.evaluate(X,Y_encoded)[0],model.evaluate(X,Y_encoded)[1]))