# %% kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% veri yukleme
data = pd.read_csv('satislar.csv')

# %% veri on isleme
aylar = data[['Aylar']]
print(aylar)

satislar = data[['Satislar']]
print(satislar)

# %% verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

"""
# %% verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""
# %% Linear Regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

prediction = lr.predict(x_test)

# %% Visualization

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
plt.title("Aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")

# %%

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, prediction)
