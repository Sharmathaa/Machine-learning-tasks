import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("D:\Dataset\mtcars.csv")


x= dataset.iloc[:,2:-1].values
y = dataset.iloc[:,1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(x_train,y_train)





regress.score(x_train,y_train)

y_predict= regress.predict(x_test)

y_predict

df = pd.DataFrame({"Actual Value":y_test,"predicted Value":y_predict})
print(df)




















