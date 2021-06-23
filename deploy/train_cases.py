# IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATASET_PATH = 'data/covid_data.csv'
MODEL_CASES_PATH = 'model_pkl/model_cases.pkl'


# IMPORTING THE DATASET
data = pd.read_csv(DATASET_PATH)

# SEGMENTING THE DATA
x = data.iloc[:,1].values
y1 = data.iloc[:,2].values


# RESHAPING THE DATA
x = np.reshape(x, (-1,1))
y1 = np.reshape(y1, (-1,1))

# INITIALIZING THE SCALERS
x_sc = StandardScaler()
y1_sc = StandardScaler()

x_scaled = x_sc.fit_transform(x)
y1_scaled = y1_sc.fit_transform(y1)

# TRAINING AND VALIDATION DATA SPLIT (DAYS VS CASES)
x1_train_sc, x1_val_sc, y1_train_sc, y1_val_sc = train_test_split(x_scaled, y1_scaled, test_size = 0.2, random_state = 0)

def model_cases():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2, input_dim = 1, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(79, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1,activation = 'linear'))
    return model



model_c = model_cases()


model_c.compile(loss = 'huber', optimizer = 'adamax', metrics = ['mse'])


model_c.fit(x1_train_sc, y1_train_sc, epochs = 500, batch_size = 150, verbose = 1, validation_split = 0.1)


pickle.dump(model_c, open(MODEL_CASES_PATH, 'wb'))
