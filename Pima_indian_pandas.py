from multiprocessing.spawn import import_main_path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd

# 1. Data loading
df = pd.read_csv("C:/Users/user2/Desktop/Gihyeon/dataset/pima-indians-diabetes.csv",names = ["pregnant","plasma","pressure","thickness","insulin","BMI","pedigree", "age", "class"])

# # 2. Data check
# #처음 3줄, 끝 3줄 보기
# print(df.head(3))
# print(df.tail(3))

# # 데이터의 전반적 정보를 확인
# print(df.info())

# # 각 정보별 특징을 좀 더 자세히 출력
# print(df.describe())

# print(df[["pregnant","plasma","pressure","thickness","insulin","BMI","pedigree", "age", "class"]])

X = df[["pregnant","plasma","pressure","thickness","insulin","BMI","pedigree", "age"]]
Y = df[["class"]]



model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(X,Y, epochs = 500, batch_size = 100)

print("\n loss: %.4f \n Accuracy: %.4f"%(model.evaluate(X,Y)[0],model.evaluate(X,Y)[1]))

# 3. Data analyse

# import matplotlib.pyplot as plt     # 그래프 그리기
# import seaborn as sns               # 


