# COVID19 PREDICTION IN INDIA

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar
# TOPICS: Regression, Machine Learning, TensorFlow

# IMPORTING REQUIRED MODULES
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1)
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('data/covid_data.csv')

x = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values

x_train,x_val,y_train,y_val = train_test_split(x,y,test_size = 0.2,random_state = 0)

x_train=np.reshape(x_train, (-1,1))
x_val=np.reshape(x_val, (-1,1))
y_train=np.reshape(y_train, (-1,1))
y_val=np.reshape(y_val, (-1,1))

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

print(scaler_x.fit(x_train))
xtrain_scale=scaler_x.transform(x_train)

print(scaler_x.fit(x_val))
xval_scale=scaler_x.transform(x_val)

print(scaler_y.fit(y_train))
ytrain_scale=scaler_y.transform(y_train)

print(scaler_y.fit(y_val))
yval_scale=scaler_y.transform(y_val)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(2, input_dim=1, kernel_initializer='normal', activation='relu'))
model.add(tf.keras.layers.Dense(472, activation = 'relu'))
model.add(tf.keras.layers.Dense(1,activation = 'linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history=model.fit(xtrain_scale, ytrain_scale, epochs=200, batch_size=150, verbose=1, validation_split=0.2)
predictions = model.predict(xval_scale)