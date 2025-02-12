import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("D:\\ABE-Neura_dataset\\time_series_data_human_activities.csv")
dataset["activity"].unique()
dataset["activity"] = dataset["activity"].map({"Walking":0,"Standing":1,"Jogging":2,"Upstairs":3,"Downstairs":4,"Sitting":5})
x = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300,random_state=0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
y_pred
df = pd.DataFrame({"predicted value":y_pred,"Actual value":y_test})
activity_mapping = {"Walking": 0, "Standing": 1, "Jogging": 2, "Upstairs": 3, "Downstairs": 4, "Sitting": 5}
reverse_mapping = {v: k for k, v in activity_mapping.items()} 
df["predicted activity"] = df["predicted value"].map(reverse_mapping)
df["actual activity"] = df["Actual value"].map(reverse_mapping)
print(df)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("Accuracy:",acc)

from sklearn.metrics import classification_report
cl_report = classification_report(y_test, y_pred)
cl_report
