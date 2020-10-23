# TESTING FILE

# IMPORTING REQUIRED MODULES
import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime

# IMPORTING THE DATASET
data = pd.read_csv("data/covid_data.csv")

# SEGMENTING THE DATA
x = data.iloc[:,1:2].values
y1 = data.iloc[:,2].values
y2 = data.iloc[:,3].values

# DEFINING THE TRAINED MODEL
model1 = pickle.load(open("model/covid_model_cases.pkl", 'rb'))
model2 = pickle.load(open("model/covid_model_deaths.pkl", 'rb'))
poly1 = PolynomialFeatures(degree = 4)
poly2 = PolynomialFeatures(degree = 4)

# PREDICTING VALUES USING TRAINED MODEL
fn_date = str(input("Enter the date to be predicted in the format DD-MM-YYYY: "))
date1 = datetime(2020,3,1)
date2 = datetime(int(fn_date[6:10]),int(fn_date[3:5]),int(fn_date[0:2]))
diff = (date2-date1).days

res1 = model1.predict(poly1.fit_transform([[diff]]))
res2 = model2.predict(poly2.fit_transform([[diff]]))

print("The number of cases in day {} is {}".format(date2,int(res1)))
print("The number of deaths in day {} is {}".format(date2,int(res2)))