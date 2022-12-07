from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# data 입력

df = pd.read_csv('C:/Users/user2/Desktop/Gihyeon/dataset/iris.csv',names = ["sepal_length","sepal_width", "petal_length","petal_width","species"])

# Data 확인

print(df.info())
print(df.describe())

# 그래프로 확인

sns.pairplot(df,hue='species')
plt.show()

# Daya 분류

X = df[["sepal_length","sepal_width", "petal_length","petal_width"]]
Y = df[["species"]]


# 문자열을 숫자로 변환 (원 핫 인코딩)

Y_encoded = pd.get_dummies(Y)

# 모델 설정

model = Sequential()
model.add(Dense(16,input_dim = 4, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 모델 컴파일

model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# 모델 실행

model.fit(X,Y_encoded,epochs=50,batch_size=1)


# 결과 출력

print("\nloss : %.4f\nAccuracy%.4f" %(model.evaluate(X,Y_encoded)[0],model.evaluate(X,Y_encoded)[1]))
