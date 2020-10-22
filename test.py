# TESTING FILE

# IMPORTING REQUIRED MODULES
import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures

# IMPORTING THE DATASET
data = pd.read_csv("data/covid_data.csv")

# SEGMENTING THE DATA
x = data.iloc[:,1:2].values
y = data.iloc[:,2].values

# DEFINING THE TRAINED MODEL
model = pickle.load(open("model/covid_model.pkl", 'rb'))
poly = PolynomialFeatures(degree = 4)

# PREDICTING VALUES USING TRAINED MODEL
pred_val = int(input("Enter the day number to be predicted: "))
res = model.predict(poly.fit_transform([[pred_val]]))
print("The number of cases in day {} is {}".format(pred_val,int(res)))