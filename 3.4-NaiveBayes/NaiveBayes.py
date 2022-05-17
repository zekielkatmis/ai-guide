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

# %% Naive Bayes
"""
Tahmin edeceğimiz sınıf;
Sürekli bir değer ise Gaussian Naive Bayes
(iki değer arasında sonsuz sayıda değere sahip değişkenler)

Nominal değere sahip veri ise Multinomial Naive Bayes
(araba markası, okul gibi verilerin tahmini için verilen değer)

Binomial değere sahip ise Benoulli Naive Bayes kullanılır
(0 veya 1 olan veri)
"""
from sklearn.naive_bayes import GaussianNB, BernoulliNB

gnb = GaussianNB()
bnn = BernoulliNB()
gnb.fit(X_train, y_train.values.ravel())
bnn.fit(X_train, y_train.values.ravel())

y_pred = gnb.predict(X_test)
y_pred2 = bnn.predict(X_test)

# %% Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

with open("output.txt", "a") as f:
    print('Confusion matrix Gaussian Naive Bayes:', file=f)
    print(cm, file=f)
    print("", file=f)

cm2 = confusion_matrix(y_test, y_pred2)

with open("output.txt", "a") as f:
    print('Confusion matrix Benoulli Naive Bayes:', file=f)
    print(cm2, file=f)