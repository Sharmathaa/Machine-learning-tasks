import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("D:\Dataset\mtcars.csv")
x = data.iloc[:,[1,9]].values
y = data.iloc[:,9].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)


from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train, y_train)
predict = regressor.predict(x_test)
 
score = regressor.score(x_train,y_train)
score

from sklearn.metrics import confusion_matrix
cn = confusion_matrix(y_test,predict)
print("confusion matrix",cn)
accuracy = (sum(np.diag(cn))/len(y_test))
print("Accuracy of how much the model gets trained",accuracy)

predicted_type = pd.DataFrame({"Actual Value":y_test,"Predicted Value":predict})
print(predicted_type)
