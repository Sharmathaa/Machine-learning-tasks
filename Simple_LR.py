import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv("D:\Dataset\Salary_Data.csv")
x = dataset.iloc[:,[0]].values
y = dataset.iloc[:,[-1]].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state= 0)


from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(x_train,y_train)
print(regress.intercept_)

print(regress.coef_)
#y_pred = 26986.69131674 + (9379.71049195 * x)r

predict = regress.predict(x_test)
predict


regress.score(x_train, y_train)


plt.figure(dpi=400)
plt.scatter(x_train,y_train)
plt.plot(x_train,regress.predict(x_train),color="purple")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()