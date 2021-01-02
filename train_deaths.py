# COVID19 PREDICTION IN INDIA

# FILE NAME: train_deaths.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Regression, Machine Learning, TensorFlow

# DISABLE TENSORFLOW DEBUG INFORMATION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print("TensorFlow Debugging Information is hidden.")

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")

DATASET_PATH = 'data/covid_data.csv'
MODEL_PATH = './model/model_deaths'

# DATA PREPROCESSING 
dataset = pd.read_csv(DATASET_PATH)
x = dataset.iloc[:,1].values
y = dataset.iloc[:,3].values
print(dataset.describe())

# RESHAPING THE DATA
x = np.reshape(x, (-1,1))
y = np.reshape(y, (-1,1))

# DEFINING THE SCALERS
x_sc = StandardScaler()
y_sc = StandardScaler()

# TRAINING AND VALIDATION DATA SPLIT
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)

# TRANSFORMING TO SCALED INPUTS
xtrain_sc = x_sc.fit_transform(x_train)
xval_sc = x_sc.transform(x_val)
ytrain_sc = y_sc.fit_transform(y_train)

# DEFINING NEURAL NETWORK FUNCTION
def model_deaths():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2, input_dim = 1, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1,activation = 'linear'))
    return model

# TRAINING THE MODEL
model = model_deaths()
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
history = model.fit(xtrain_sc, ytrain_sc, epochs = 500, batch_size = 150, verbose = 1, validation_split = 0.2)
print("Model Trained Successfully.")

# PLOTTING THE LOSS GRAPH
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
plt.show()
plt.savefig('graphs/deaths_loss_graph.png')

# CALCULATING THE ACCURACY
ypred_sc = model.predict(xval_sc)
y_pred = y_sc.inverse_transform(ypred_sc)
print(f"Model Accuracy: {round(r2_score(y_val,y_pred)*100,4)}")

# SAVING THE MODEL
tf.keras.models.save_model(model,MODEL_PATH)
print(f"Successfully stored the trained model at {MODEL_PATH}")