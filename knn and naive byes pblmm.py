import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("D:\Dataset\wbcd.csv")
dataset['diagnosis'] = dataset['diagnosis'].map({'B': 0, 'M' : 1})
x = dataset.iloc[:,2:-1].values
y = dataset.iloc[:,1].values






from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


from sklearn.naive_bayes import MultinomialNB
classify = MultinomialNB()
classify.fit(x_train, y_train)
 
pred = classify.predict(x_test)
pred
pred_df = pd.DataFrame({"actual value":y_test,"Predicted ouput":pred})
pred_df
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, pred)
mat
acc = (sum(np.diag(mat))/len(y_test))
acc









# KNN classification code
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("D:\Dataset\wbcd.csv")
dataset['diagnosis'] = dataset['diagnosis'].map({'B': 0, 'M' : 1})
x = dataset.iloc[:,2:-1].values
y = dataset.iloc[:,1].values



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)



from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=6,metric="minkowski",p=2)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

acc = (sum(np.diag(cm))/len(y_test))
acc

























