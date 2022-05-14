import numpy as np
import matplotlib.pyplot as plt
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

"""
fit_transform => standart scaler'ı x_train ile eğit nasıl scal edeceğini öğret
transform => x_test için tekrar eğitme x_trainden öğrendiği gibi sonuçlar çıkar
"""

# %% Logistic Regression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)

# %% Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

with open("output.txt", "a") as f:
    print('Confusion matrix:', file=f)
    print(cm, file=f)
    print("", file=f)
"""
[0 1]
[7 0]
1 tane erkek değerinden 0 doğru 1 yanlış tahmin edilmiş
7 tane kiz değerinden 7 yanlış 0 doğru tahmin edilmiş
"""

# %% outlier detection
plt.scatter(x['boy'], x["kilo"])

new_x = x.drop(labels=[0,1,2,3,4], axis=0)
new_x = new_x.reset_index()
new_x = new_x.drop("index", axis=1)

new_y = y.drop(labels=range(0, 5), axis=0)
new_y = new_y.reset_index()
new_y = new_y.drop("index", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    new_x, new_y, test_size=0.33, random_state=0)

X_train = sc.fit_transform(x_train)
X_test =  sc.transform(x_test)

logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
with open("output.txt", "a") as f:
    print('Confusion matrix without outlier:', file=f)
    print(cm, file=f)
"""
[3 0]
[0 3]
3 tane erkek değerinden 3 doğru 0 yanlış tahmin edilmiş
3 tane kiz değerinden 0 yanlış 3 doğru tahmin edilmiş

outlier'lar çıkınca yüzde yüz doğru tahmin etmiş olduk
"""
