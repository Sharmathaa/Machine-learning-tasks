import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("D:\Dataset\wine_customer.csv")
x = dataset.iloc[:,[0,-1]].values

from sklearn.cluster import KMeans
wcss = [ ]
for i in range(1,18):
    k_means = KMeans(n_clusters=i,init='k-means++',random_state=0)
    k_means.fit(x)
    wcss.append(k_means.inertia_)
    
plt.figure(dpi=400)
plt.plot(range(1,18),wcss)
plt.title("elbow method")
plt.xlabel("number of clusters")
plt.ylabel("wcss")
plt.show()


k_means = KMeans(n_clusters=5,random_state=0)
y_kmeans = k_means.fit_predict(x)

k_means.labels_
k_means.cluster_centers_



plt.figure(dpi=400)
plt.scatter(x[:,0],x[:,1],c="black")
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=10,c="red",label="customer type 1")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=10,c="blue",label="customer type 2")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=10,c="green",label="customer type 3")
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=10,c="cyan",label="customer type 4")
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=10,c="purple",label="customer type 5")
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=300,c="orange",label="centroids")
plt.title("cluster of customer")
plt.legend()
plt.show()





# hirerarchial clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("D:\Dataset\wine_customer.csv")
x = dataset.iloc[:,[0,-1]].values

from scipy.cluster import hierarchy as sch
plt.figure(dpi=400)
dendogram = sch.dendrogram(sch.linkage(x,method="complete"))
plt.title("DENDOGRAM")
plt.show()

from sklearn.cluster import AgglomerativeClustering
clu = AgglomerativeClustering(n_clusters=2,linkage="complete",affinity="euclidean")
y_pred = clu.fit_predict(x)

plt.figure(dpi=400)
plt.scatter(x[y_pred==0,0],x[y_pred==0,1],s=10,c="red",label="customer type 1")
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],s=10,c="blue",label="customer type 2")
plt.legend()
plt.title("cluster of customer")
plt.show()


























    
    
    