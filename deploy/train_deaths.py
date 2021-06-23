# IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATASET_PATH = 'data/covid_data.csv'
MODEL_DEATHS_PATH = 'model_pkl/model_deaths.pkl'

# IMPORTING THE DATASET
data = pd.read_csv(DATASET_PATH)

# SEGMENTING THE DATA
x = data.iloc[:,1].values
y2 = data.iloc[:,3].values

# RESHAPING THE DATA
x = np.reshape(x, (-1,1))
y2 = np.reshape(y2, (-1,1))

# INITIALIZING THE SCALERS
x_sc = StandardScaler()
y2_sc = StandardScaler()

x_scaled = x_sc.fit_transform(x)
y2_scaled = y2_sc.fit_transform(y2)

# TRAINING AND VALIDATION DATA SPLIT (DAYS VS DEATHS)
x2_train_sc, x2_val_sc, y2_train_sc, y2_val_sc = train_test_split(x_scaled, y2_scaled, test_size = 0.2, random_state = 0)

def model_deaths():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2, input_dim = 1, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1,activation = 'linear'))
    return model

model_d = model_deaths()

model_d.compile(loss = 'huber', optimizer = 'adamax', metrics = ['mse'])
model_d.fit(x2_train_sc, y2_train_sc, epochs = 500, batch_size = 150, verbose = 1, validation_split = 0.1)
pickle.dump(model_d, open(MODEL_DEATHS_PATH, 'wb'))