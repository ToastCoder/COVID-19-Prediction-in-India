# COVID19 PREDICTION IN INDIA

# FILE NAME: test.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Regression, Machine Learning, TensorFlow

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# SCALING THE DATA
x_sc = StandardScaler()
y1_sc = StandardScaler()
y2_sc = StandardScaler()

x_sc.fit(x)
y1_sc.fit(y1)
y2_sc.fit(y2)

# DEFINING THE TRAINED MODEL
model_c = tf.keras.models.load_model("model/model_cases", custom_objects=None, compile=True)
model_d = tf.keras.models.load_model("model/model_deaths", custom_objects=None, compile=True)

# PREDICTING VALUES USING TRAINED MODEL
fn_date = str(input("Enter the date to be predicted in the format DD-MM-YYYY: "))
date1 = datetime(2020,3,2)
date2 = datetime(int(fn_date[6:10]),int(fn_date[3:5]),int(fn_date[0:2]))
diff = (date2-date1).days

diff = np.array(diff)
diff = np.reshape(diff, (-1,1))
diff_sc = x_sc.transform(diff)

res1_sc = model_c.predict(diff_sc)
res2_sc = model_d.predict(diff_sc)

res1 = y1_sc.inverse_transform(res1_sc)
res2 = y2_sc.inverse_transform(res2_sc)

print(f"The estimated number of cases in day {date2.strftime('%d-%m-%Y')} is {'{:,}'.format(int(res1))}")
print(f"The estimated number of deaths in day {date2.strftime('%d-%m-%Y')} is {'{:,}'.format(int(res2))}")