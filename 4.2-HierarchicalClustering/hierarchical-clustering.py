import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% read data
data = pd.read_csv("musteriler.csv")

x = data[["Hacim", "Maas"]]

# %% hierarchical clustering
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=4, 
                             affinity="euclidean", 
                             linkage='ward')

"""
affinity: Metric used to compute the linkage. 
"euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"

linkage: which distance to use between sets of observation.
'ward', 'complete', 'average', 'single'

if (linkage="ward") => affinity="euclidean"
"""

y_pred = ac.fit_predict(x)

with open("output.txt", "a") as f:
    print('Agglomerative Clustering:', file=f)
    print('Prediction of which cluster the data will be in:', file=f)
    print(y_pred, file=f)

# %% visualize
X = x.to_numpy()
plt.scatter(X[y_pred==0,0], X[y_pred==0,1],s=100, c="red")
plt.scatter(X[y_pred==1,0], X[y_pred==1,1],s=100, c="blue")
plt.scatter(X[y_pred==2,0], X[y_pred==2,1],s=100, c="green")
plt.scatter(X[y_pred==3,0], X[y_pred==3,1],s=100, c="yellow")
plt.show()

# %% dendrogram
import scipy.cluster.hierarchy as sch
from matplotlib.pyplot import figure

figure(figsize=(64, 48), dpi=80)
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.show()
