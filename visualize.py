# COVID19 PREDICTION IN INDIA

# FILE NAME: visualize.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Regression, Machine Learning, TensorFlow

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# IMPORTING THE DATASET
data = pd.read_csv("data/covid_data.csv")

# SEGMENTING THE DATA
x = data.iloc[:,1].values
y1 = data.iloc[:,2].values
y2 = data.iloc[:,3].values

# RESHAPING THE DATA
x = np.reshape(x, (-1,1))
y1 = np.reshape(y1, (-1,1))
y2 = np.reshape(y2, (-1,1))

# TRAINING AND VALIDATION DATA SPLIT (DAYS VS CASES)
x1_train, x1_val, y1_train, y1_val = train_test_split(x, y1, test_size = 0.2, random_state = 0)

# RESHAPING THE DATA (DAYS VS CASES)
x1_train = np.reshape(x1_train, (-1,1))
x1_val = np.reshape(x1_val, (-1,1))
y1_train = np.reshape(y1_train, (-1,1))
y1_val = np.reshape(y1_val, (-1,1))

# SCALING THE DATA (DAYS VS CASES)
scaler_x1 = MinMaxScaler()
scaler_y1 = MinMaxScaler()

x1train_scaled = scaler_x1.fit(x1_train)
y1train_scaled = scaler_y1.fit(y1_train)

# TRAINING AND VALIDATION DATA SPLIT (DAYS VS DEATHS)
x2_train, x2_val, y2_train, y2_val = train_test_split(x, y2, test_size = 0.2, random_state = 0)

# RESHAPING THE DATA (DAYS VS DEATHS)
x2_train = np.reshape(x2_train, (-1,1))
x2_val = np.reshape(x2_val, (-1,1))
y2_train = np.reshape(y2_train, (-1,1))
y2_val = np.reshape(y2_val, (-1,1))

# SCALING THE DATA (DAYS VS DEATHS)
scaler_x2 = MinMaxScaler()
scaler_y2 = MinMaxScaler()

x2train_scaled = scaler_x2.fit_transform(x2_train)
x2val_scaled = scaler_x2.fit_transform(x2_val)
y2train_scaled = scaler_y2.fit_transform(y2_train)
y2val_scaled = scaler_y2.fit_transform(y2_val)

x_scaled = scaler_x1.transform(x)
y1_scaled = scaler_y1.transform(y1)
y2_scaled = scaler_y2.transform(y2)

# DEFINING THE TRAINED MODEL
model_c = load_model("model/model_cases",custom_objects=None,compile=True)
model_d = load_model("model/model_deaths",custom_objects=None,compile=True)

# CALCULATING THE ESTIMATED ACCURACY (DAYS VS CASES)
y1pred_scaled = model_c.predict(x1val_scaled)
y1_pred = scaler_y1.inverse_transform(y1pred_scaled)
print(f"The estimated accuracy of the model_c (Days vs Cases) is: {round(r2_score(y1_val,y1_pred)*100,4)}")

# CALCULATING THE ESTIMATED ACCURACY (DAYS VS DEATHS)
y2pred_scaled = model_d.predict(x2val_scaled)
y2_pred = scaler_y2.inverse_transform(y2pred_scaled)
print(f"The estimated accuracy of the model_d (Days vs Deaths) is: {round(r2_score(y2_val,y2_pred)*100,4)}")

y1_est = model_c.predict(x_scaled)
y1_est = scaler_y1.inverse_transform(y1_est)

y2_est = model_d.predict(x_scaled)
y2_est = scaler_y2.inverse_transform(y2_est)

# VISUALIZATION OF ACTUAL DATA TO PREDICTED DATA (DAYS VS CASES)
plt.plot(x,y1, color = 'blue', label ='Actual Data' )
plt.plot(x,y1_est, color = 'red', label = 'Predicted Data')
plt.title('COVID-19 Prediction (Days vs Cases')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.legend()
plt.show()

# VISUALIZATION OF ACTUAL DATA TO PREDICTED DATA (DAYS VS DEATHS)
plt.plot(x,y2, color = 'blue', label ='Actual Data' )
plt.plot(x,y2_est, color = 'red', label = 'Predicted Data')
plt.title('COVID-19 Prediction (Days vs Deaths')
plt.xlabel('Days')
plt.ylabel('Deaths')
plt.legend()
plt.show()

