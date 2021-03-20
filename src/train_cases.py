#-------------------------------------------------------------------------------------------------------------------------------

# COVID19 PREDICTION IN INDIA

# FILE NAME: train_cases.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Regression, Machine Learning, TensorFlow

#-------------------------------------------------------------------------------------------------------------------------------

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import argparse

# FUNCTION FOR PARSING ARGUMENTS
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epochs',type = int, default = 500, required = False)
    parser.add_argument('-bs', '--batch_size',type = int, default = 150, required = False)
    parser.add_argument('-l','--loss',type = str, default = 'huber', required = False)
    parser.add_argument('-op','--optimizer', type = str, default = 'adamax', required = False)
    args = parser.parse_args()
    return args

plt.style.use('ggplot')

print(f"TensorFlow version: {tf.__version__}")

DATASET_PATH = 'data/covid_data.csv'
MODEL_PATH = './model/model_cases'
args = parse()

# DATA PREPROCESSING 
dataset = pd.read_csv(DATASET_PATH)
x = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values
print("Dataset Description:\n",dataset.describe())
print("Dataset Head\n",dataset.head())

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
def model_cases():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2, input_dim = 1, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1,activation = 'linear'))
    return model

# TRAINING THE MODEL
model = model_cases()
model.compile(loss = args.loss, optimizer = args.optimizer, metrics = ['mse'])
history = model.fit(xtrain_sc, ytrain_sc, epochs = args.epochs, batch_size = args.batch_size, verbose = 1, validation_split = 0.1)
print("Model Trained Successfully.")

# PLOTTING THE LOSS GRAPH
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
plt.savefig('graphs/cases_loss_graph.png')
plt.show()

# CALCULATING THE ACCURACY
ypred_sc = model.predict(xval_sc)
y_pred = y_sc.inverse_transform(ypred_sc)
print(f"Model Accuracy: {round(r2_score(y_val,y_pred)*100,4)}")

# SAVING THE MODEL
tf.keras.models.save_model(model,MODEL_PATH)
print(f"Successfully stored the trained model at {MODEL_PATH}")