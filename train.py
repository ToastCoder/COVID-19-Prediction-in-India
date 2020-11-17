# COVID19 PREDICTION IN INDIA

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar
# TOPICS: Regression, Random Forest Regression, Machine Learning

# IMPORTING REQUIRED MODULES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# READING DATASET
data = pd.read_csv("data/covid_data.csv")

# SEGMENTING DATA
x = data.iloc[:,1:2].values
y_cases = data.iloc[:,2].values
y_deaths = data.iloc[:,3].values

# CREATING AND TRAINING THE MODELS

# MODEL FOR PREDICTING CASES
rfr1 = RandomForestRegressor(n_estimators = 100)
rfr1.fit(x,y_cases)

# MODEL FOR PREDICTING DEATHS
rfr2 = RandomForestRegressor(n_estimators = 100)
rfr2.fit(x,y_deaths)

# PLOTTING DATA

# PLOT FOR VISUALIZING CASES
plt.scatter(x,y_cases, color = 'blue')
plt.plot(x, rfr1.predict(x), color = 'red')
plt.title('Covid-19 Cases (Days vs Cases)')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.show()

# PLOT FOR VISUALIZING DEATHS
plt.scatter(x,y_deaths, color = 'blue')
plt.plot(x, rfr2.predict(x), color = 'red')
plt.title('Covid-19 Cases (Days vs Deaths)')
plt.xlabel('Days')
plt.ylabel('Deaths')
plt.show()

# CALCULATING ACCURACY OF THE TWO MODELS
print(f"The estimated accuracy for model for predicting cases is {round(r2_score(y_cases,rfr1.predict(x))*100,4)} %")
print(f"The estimated accuracy for model for predicting deaths is {round(r2_score(y_deaths,rfr2.predict(x))*100,4)} %")

# SAVING THE MODEL
pickle.dump(rfr1,open("model/covid_model_cases.pkl","wb"))
pickle.dump(rfr2,open("model/covid_model_deaths.pkl","wb"))
