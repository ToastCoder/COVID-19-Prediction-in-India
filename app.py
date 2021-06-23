# IMPORTING REQUIRED PACKAGES
import requests
from flask import Flask, render_template, redirect, url_for, request,jsonify
from werkzeug.wrappers import Request, Response
import json
import numpy as np 
import pandas as pd
from scipy import stats
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import warnings
from datetime import datetime

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=PendingDeprecationWarning)

DATASET_PATH = "data/covid_data.csv"

# IMPORTING THE DATASET
data = pd.read_csv(DATASET_PATH)

# SEGMENTING THE DATA
x = data.iloc[:,1].values
y1 = data.iloc[:,2].values
y2 = data.iloc[:,3].values

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

app=Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# STARTING INDEX PAGE
@app.route("/")
def index():
   	return render_template("index.html")

# RETURN MODEL PREDICTION
@app.route("/api/predict", methods=["GET"])
def predict():
    msg_data={}
    for k in request.args.keys():
        val=request.args.get(k)
        msg_data[k]=val
    date1 = datetime(2020,3,2)
    date2 = datetime(int(msg_data['year']),int(msg_data['month']),int(msg_data['date']))
    diff = (date2-date1).days
    diff = np.array(diff)
    diff = np.reshape(diff, (-1,1))
    diff_sc = x_sc.transform(diff)
    model_c = tf.keras.models.load_model("model/model_cases",custom_objects=None,compile=True)
    model_d = tf.keras.models.load_model("model/model_deaths",custom_objects=None,compile=True)
    res1 = model_c.predict(diff_sc)
    res2 = model_d.predict(diff_sc)
    res = [res1,res2]
    return res

if __name__ == "__main_":
	app.debug = False
	from werkzeug.serving import run_simple
	run_simple("localhost", 5000, app)