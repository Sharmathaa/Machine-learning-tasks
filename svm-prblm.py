import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("D:\Dataset\Wine.csv")
x = dataset.iloc[:,[11,12]].values
y = dataset.iloc[:,[-1]].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.svm import SVC
cf = SVC(C=1.0, kernel='rbf',gamma=0.95,random_state=0)
cf.fit(x_train,y_train)
#y_test = np.reshape(y_test,(-1,1))
y_pred = cf.predict(x_test)


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
cm
acc = accuracy_score(y_test, y_pred)
acc
from matplotlib.colors import ListedColormap
x1,x2= np.meshgrid(np.arange(start=x_train[:,0].min()-1,stop=x_train[:,0].max()-1,step=0.01),
                   np.arange(start=x_train[:,-1].min()-1,stop=x_train[:,-1].max()+1,step=0.01))   
plt.contourf(x1,x2,cf.predict(np.array([np.ravel(x1),np.ravel(x2)]).T).reshape(x1.shape),alpha=0.5,cmap=ListedColormap(("yellow","blue")))     


