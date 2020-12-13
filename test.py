# COVID19 PREDICTION IN INDIA

# FILE NAME: test.py

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

# IMPORTING THE DATASET
data = pd.read_csv("data/covid_data.csv")

# SEGMENTING THE DATA
x = data.iloc[:,1].values
y1 = data.iloc[:,2].values
y2 = data.iloc[:,3].values

# DEFINING THE TRAINED MODEL
model_c = load_model("model/model_cases.h5")
model_d = load_model("model/model_deaths.h5")

# PREDICTING VALUES USING TRAINED MODEL
fn_date = str(input("Enter the date to be predicted in the format DD-MM-YYYY: "))
date1 = datetime(2020,3,1)
date2 = datetime(int(fn_date[6:10]),int(fn_date[3:5]),int(fn_date[0:2]))
diff = (date2-date1).days

diff = np.array(diff,dtype = int)
diff = np.reshape(diff, (-1,1))

scaler = MinMaxScaler()

diff_scaled = scaler.fit_transform(diff)

res1_scaled = model_c.predict(diff_scaled)
res2_scaled = model_d.predict(diff_scaled)

res1 = scaler.inverse_transform(res1_scaled)
res2 = scaler.inverse_transform(res2_scaled)

print("The estimated number of cases in day {} is {}".format(date2,int(res1)))
print("The estimated number of deaths in day {} is {}".format(date2,int(res2)))

print(diff)