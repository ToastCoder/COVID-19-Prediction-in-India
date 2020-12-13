# COVID19 PREDICTION IN INDIA

# FILE NAME: train_cases.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Regression, Machine Learning, TensorFlow

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# DATA PREPROCESSING 
dataset = pd.read_csv('data/covid_data.csv')
x = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values

# TRAINING AND VALIDATION DATA SPLIT
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size = 0.2,random_state = 0)

# RESHAPING THE DATA
x_train = np.reshape(x_train, (-1,1))
x_val = np.reshape(x_val, (-1,1))
y_train = np.reshape(y_train, (-1,1))
y_val = np.reshape(y_val, (-1,1))

# SCALING THE DATA
scaler = MinMaxScaler()
xtrain_scaled = scaler.fit_transform(x_train)
xval_scaled = scaler.fit_transform(x_val)
ytrain_scaled = scaler.fit_transform(y_train)
yval_scaled = scaler.fit_transform(y_val)

# DEFINING NEURAL NETWORK AND ITS LAYERS
model = Sequential()
model.add(Dense(2, input_dim=1, kernel_initializer='normal', activation='relu'))
model.add(Dense(472, activation = 'relu'))
model.add(Dense(1,activation = 'linear'))

# TRAINING THE MODEL
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history=model.fit(xtrain_scaled, ytrain_scaled, epochs=500, batch_size=150, verbose=1, validation_split=0.2)
predictions = model.predict(xval_scaled)

# PLOTTING THE GRAPH FOR TRAIN-LOSS AND VALIDATION-LOSS
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# CALCULATING THE ESTIMATED ACCURACY
ypred_scaled = model.predict(xval_scaled)
y_pred = scaler.inverse_transform(ypred_scaled)
print(f"The estimated accuracy of the model is: {round(r2_score(y_val,y_pred)*100,4)}")

# SAVING THE MODEL
model.save('model/model_cases')
