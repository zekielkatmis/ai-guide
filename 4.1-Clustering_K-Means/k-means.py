import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% read data
data = pd.read_csv("musteriler.csv")

x = data[["Hacim", "Maas"]]

# %% k-means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, init="k-means++")
y_pred = kmeans.fit_predict(x)

X = x.to_numpy()
plt.scatter(X[y_pred==0,0], X[y_pred==0,1],s=100, c="red")
plt.scatter(X[y_pred==1,0], X[y_pred==1,1],s=100, c="blue")
plt.scatter(X[y_pred==2,0], X[y_pred==2,1],s=100, c="green")
plt.scatter(X[y_pred==3,0], X[y_pred==3,1],s=100, c="yellow")
plt.show()

with open("output.txt", "a") as f:
    print('K-Means cluster points:', file=f)
    print(kmeans.cluster_centers_, file=f)

# %% WCSS ( Within-Cluster Sum of Square )
results=[]

for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state = 123)
    kmeans.fit(x)
    results.append(kmeans.inertia_)
    
plt.plot(range(1,10), results)
plt.show()

"""
tablodaki kırılımlara bakıcak olursak çıkan wcss değerlerine göre 
en iyi n_clusters değeri bu veri için 3 veya 4 olabilir
"""