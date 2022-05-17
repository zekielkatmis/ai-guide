import numpy as np
import pandas as pd

# %% data read
data = pd.read_csv('veriler.csv')

x = data.drop(["ulke", "cinsiyet"], axis=1) # data.iloc[:,1:4].values
y = data[["cinsiyet"]] # data.iloc[:,4:].values

# %% verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=0)

# %% Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test =  sc.transform(x_test)

# %% KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
knn.fit(X_train, y_train.values.ravel())

y_pred = knn.predict(X_test)

# %% Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

with open("output.txt", "a") as f:
    print('Confusion matrix when n_neighbors=5:', file=f)
    print(cm, file=f)
    print("", file=f)

# %% lower n_neighbors
"""
bakılacak komşu sayısını düşürmek aslında o kadar kötü bir durum değildir
bu sayede makine çok daha fazla veriyi inceleyip karışıklığı önleyebilecek
bu örnekte görüldüğü gibi bakılacak komşu sayısı azalınca
outlierlar olsa bile daha iyi tahmin yapan bir modelimiz olacak
"""
knn2 = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
knn2.fit(X_train, y_train.values.ravel())

y_pred2 = knn2.predict(X_test)

cm2 = confusion_matrix(y_test, y_pred2)
with open("output.txt", "a") as f:
    print('Confusion matrix when n_neighbors=1:', file=f)
    print(cm2, file=f)
    print("", file=f)