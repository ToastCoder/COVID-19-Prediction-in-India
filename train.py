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
y1 = data.iloc[:,2].values
y2 = data.iloc[:,3].values

# FITTING THE DATA
poly1 = PolynomialFeatures(degree = 4)
poly_x1 = poly1.fit_transform(x)
poly1.fit(poly_x1,y1)
poly_model1 = LinearRegression()
poly_model1.fit(poly_x1,y1)

poly2 = PolynomialFeatures(degree = 4)
poly_x2 = poly2.fit_transform(x)
poly2.fit(poly_x2,y2)
poly_model2 = LinearRegression()
poly_model2.fit(poly_x1,y2)

# PLOTTING DATA
plt.scatter(x, y1, color = 'blue')
plt.plot(x, poly_model1.predict(poly_x1), color = 'green')
plt.title('Covid-19 Cases (Days vs Cases)')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.show()

plt.scatter(x, y2, color = 'red')
plt.plot(x, poly_model2.predict(poly_x2), color = 'purple')
plt.title('Covid-19 Cases (Days vs Deaths)')
plt.xlabel('Days')
plt.ylabel('Deaths')
plt.show()

# SAVING THE MODEL
pickle.dump(poly_model1,open("model/covid_model_cases.pkl","wb"))
pickle.dump(poly_model2,open("model/covid_model_deaths.pkl","wb"))
