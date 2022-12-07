from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('C:/Users/user2/Desktop/Gihyeon/dataset/iris.csv',names=["car_length"])