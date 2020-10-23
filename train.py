# COVID19 PREDICTION IN INDIA

# DEVELOPED BY: Vigneshwar Ravichandar
# TOPICS: Regression, Polynomial Regression, Machine Learning

# IMPORTING REQUIRED MODULES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

# READING DATASET
data = pd.read_csv("data/covid_data.csv")

# SEGMENTING DATA
x = data.iloc[:,1:2].values
y = data.iloc[:,2].values

# FITTING THE DATA
poly = PolynomialFeatures(degree = 6)
poly_x = poly.fit_transform(x)
poly.fit(poly_x,y)
poly_model = LinearRegression()
poly_model.fit(poly_x,y)

# PLOTTING DATA
plt.scatter(x, y, color = 'blue')
plt.plot(x, poly_model.predict(poly_x), color = 'red')
plt.title('Covid-19 Cases (Days vs Cases)')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.show()

# SAVING THE MODEL
pickle.dump(poly_model,open("model/covid_model.pkl","wb"))
