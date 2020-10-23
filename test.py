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
y = data.iloc[:,2].values

# DEFINING THE TRAINED MODEL
model = pickle.load(open("model/covid_model.pkl", 'rb'))
poly = PolynomialFeatures(degree = 4)

# PREDICTING VALUES USING TRAINED MODEL
fn_date = str(input("Enter the date to be predicted in the format DD-MM-YYYY: "))
date1 = datetime(2020,3,1)
date2 = datetime(int(fn_date[6:10]),int(fn_date[3:5]),int(fn_date[0:2]))
diff = (date2-date1).days
res = model.predict(poly.fit_transform([[diff]]))
print("The number of cases in day {} is {}".format(date2,int(res)))