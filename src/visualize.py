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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

plt.style.use('ggplot')

print(f"TensorFlow version: {tf.__version__}")

DATASET_PATH = "data/covid_data.csv"

# IMPORTING THE DATASET
data = pd.read_csv(DATASET_PATH)

# SEGMENTING THE DATA
x = data.iloc[:,1].values
y1 = data.iloc[:,2].values
y2 = data.iloc[:,3].values
print("Dataset Description:\n",data.describe())
print("Dataset Head\n",data.head())

# RESHAPING THE DATA
x = np.reshape(x, (-1,1))
y1 = np.reshape(y1, (-1,1))
y2 = np.reshape(y2, (-1,1))

# INITIALIZING THE SCALERS
x_sc = StandardScaler()
y1_sc = StandardScaler()
y2_sc = StandardScaler()

x_scaled = x_sc.fit_transform(x)
y1_scaled = y1_sc.fit_transform(y1)
y2_scaled = y2_sc.fit_transform(y2)

# TRAINING AND VALIDATION DATA SPLIT (DAYS VS CASES)
x1_train_sc, x1_val_sc, y1_train_sc, y1_val_sc = train_test_split(x_scaled, y1_scaled, test_size = 0.2, random_state = 0)

# TRAINING AND VALIDATION DATA SPLIT (DAYS VS DEATHS)
x2_train_sc, x2_val_sc, y2_train_sc, y2_val_sc = train_test_split(x_scaled, y2_scaled, test_size = 0.2, random_state = 0)

# DEFINING THE TRAINED MODEL
model_c = tf.keras.models.load_model("model/model_cases",custom_objects=None,compile=True)
model_d = tf.keras.models.load_model("model/model_deaths",custom_objects=None,compile=True)

# CALCULATING THE ESTIMATED ACCURACY (DAYS VS CASES)
y1_pred_sc = model_c.predict(x1_val_sc)
y1_pred = y1_sc.inverse_transform(y1_pred_sc)
y1_val = y1_sc.inverse_transform(y1_val_sc)
print(f"Accuracy of the model_c (Days vs Cases) is: {round(r2_score(y1_val,y1_pred)*100,4)}")

# CALCULATING THE ESTIMATED ACCURACY (DAYS VS DEATHS)
y2_pred_sc = model_d.predict(x2_val_sc)
y2_pred = y2_sc.inverse_transform(y2_pred_sc)
y2_val = y2_sc.inverse_transform(y2_val_sc)
print(f"Accuracy of the model_d (Days vs Deaths) is: {round(r2_score(y2_val,y2_pred)*100,4)}")

y1_est = model_c.predict(x_scaled)
y1_est = y1_sc.inverse_transform(y1_est)

y2_est = model_d.predict(x_scaled)
y2_est = y2_sc.inverse_transform(y2_est)

# VISUALIZATION OF ACTUAL DATA TO PREDICTED DATA (DAYS VS CASES)
plt.figure(0)
plt.plot(x,y1, color = 'blue', label ='Actual Data' )
plt.plot(x,y1_est, color = 'red', label = 'Predicted Data')
plt.title('COVID-19 Prediction (Days vs Cases)')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.legend()
plt.savefig('graphs/days_vs_cases_ip.png')
plt.show()

# VISUALIZATION OF ACTUAL DATA TO PREDICTED DATA (DAYS VS DEATHS)
plt.figure(1)
plt.plot(x,y2, color = 'blue', label ='Actual Data' )
plt.plot(x,y2_est, color = 'red', label = 'Predicted Data')
plt.title('COVID-19 Prediction (Days vs Deaths)')
plt.xlabel('Days')
plt.ylabel('Deaths')
plt.legend()
plt.savefig('graphs/days_vs_deaths_ip.png')
plt.show()

