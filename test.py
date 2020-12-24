# COVID19 PREDICTION IN INDIA

# FILE NAME: test.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Regression, Machine Learning, TensorFlow

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

# SCALING THE DATA
scaler_x = MinMaxScaler()
scaler_y1 = MinMaxScaler()
scaler_y2 = MinMaxScaler()

scaler_x.fit(x)
scaler_y1.fit(y1)
scaler_y2.fit(y2)

# DEFINING THE TRAINED MODEL
model_c = load_model("model/model_cases",custom_objects=None,compile=True)
model_d = load_model("model/model_deaths",custom_objects=None,compile=True)

# PREDICTING VALUES USING TRAINED MODEL
fn_date = str(input("Enter the date to be predicted in the format DD-MM-YYYY: "))
date1 = datetime(2020,3,1)
date2 = datetime(int(fn_date[6:10]),int(fn_date[3:5]),int(fn_date[0:2]))
diff = (date2-date1).days
diff = np.array(diff)
diff = np.reshape(diff, (-1,1))
diffscaled = scaler_x.transform(diff)
res1_scaled = model_c.predict(diffscaled)
res2_scaled = model_d.predict(diffscaled)
res1 = scaler_y1.inverse_transform(res1_scaled)
res2 = scaler_y2.inverse_transform(res2_scaled)
print(f"The estimated number of cases in day {date2} is {int(res1)}")
print(f"The estimated number of deaths in day {date2} is {int(res2)}")