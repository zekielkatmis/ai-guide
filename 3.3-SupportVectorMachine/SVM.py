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
from sklearn.svm import SVC

svc = SVC(kernel="sigmoid")
"""
kernel{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, 
default='rbf'
Specifies the kernel type to be used in the algorithm. 
"""
svc.fit(X_train, y_train.values.ravel())

y_pred = svc.predict(X_test)

# %% Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

with open("output.txt", "a") as f:
    print('Confusion matrix when kernel=sigmoid:', file=f)
    print(cm, file=f)
    print("", file=f)
