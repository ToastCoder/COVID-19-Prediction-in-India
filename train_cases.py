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
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt

# DATA PREPROCESSING 
dataset = pd.read_csv('data/covid_data.csv')
x = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values

# RESHAPING THE DATA
x = np.reshape(x, (-1,1))
y = np.reshape(y, (-1,1))

# TRAINING AND VALIDATION DATA SPLIT
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 0)

# SCALING THE DATA
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

xtrain_scaled = scaler_x.fit_transform(x_train)
xval_scaled = scaler_x.transform(x_val)
ytrain_scaled = scaler_y.fit_transform(y_train)

# DEFINING NEURAL NETWORK AND ITS LAYERS
model = Sequential()
model.add(Dense(2, input_dim=1, kernel_initializer='normal', activation='relu'))
model.add(Dense(79, activation = 'relu'))
model.add(Dense(1,activation = 'linear'))

# TRAINING THE MODEL
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history=model.fit(xtrain_scaled, ytrain_scaled, epochs=500, batch_size=150, verbose=1, validation_split=0.2)

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
y_pred = scaler_y.inverse_transform(ypred_scaled)
print(f"The estimated accuracy of the model is: {round(r2_score(y_val,y_pred)*100,4)}")

# SAVING THE MODEL
PATH = './model/model_cases'
save_model(model,PATH)
print(f"Successfully stored the trained model at {PATH}")
