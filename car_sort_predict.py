# -*- coding: cp949 -*-

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# 전처리
df = pd.read_csv("C:/Users/user2/Desktop/Gihyeon/dataset/carr.csv",names = ["width", "length" ,"height","baegi","yeonbi","class"], encoding='cp949')

# standardization_df = (df - df.mean())/df.std()  # 표준화
# normalization_df = (df-df.min())/(df.max()-df.min()) # 정규화

X1 = df[["width","length", "height","baegi","yeonbi"]]
normalization_df = (X1-X1.min())/(X1.max()-X1.min()) # 정규화

X = normalization_df[["width","length", "height","baegi","yeonbi"]]
Y = df[["class"]]
# 원-핫 인코딩
Y_encoded = pd.get_dummies(Y)

# 트레이닝, 테스트 셋 분리
X_train, X_test, Y_train, Y_test = train_test_split(X,Y_encoded,test_size=0.3,random_state=3)

# 입력층, 은닉층, 출력층 설정
model = Sequential()
model.add(Dense(24, input_dim = 5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))


# data compile
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# load path
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.h5py"

# check point
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',verbose=1, save_best_only=True)

# early stopping
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=30)

# model 실행
history = model.fit(X_train,Y_train, validation_data=(X_test,Y_test) , epochs=3500, batch_size= 150, verbose=0,callbacks=[early_stopping_callback,checkpointer])

# model prediction
Y_prediction = model.predict(X_test).flatten()

for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("real : {:.3f}, predict : {:.3f}".format(label,prediction))

# test loss, test accuracy 출력
print("\nloss : %.4f\nAccuracy : %.4f" %(model.evaluate(X_test,Y_test)[0],model.evaluate(X_test,Y_test)[1]))

# #그래프그리기
# y_vloss=history.history['val_loss']
# y_acc=history.history['accuracy']

# x_len=np.arange(len(y_acc))
# plt.plot(x_len,y_vloss,"-",c="red",markersize=3)
# plt.plot(x_len,y_acc,"-",c="blue",markersize=3)

# plt.ylim([0,1])

# plt.show()
