# %% kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% veri yukleme
data = pd.read_csv('maaslar.csv')

plt.scatter(data["Egitim Seviyesi"],data["maas"])
plt.title("Eğitim/Maaş")
plt.xlabel("Maaş")
plt.ylabel("Eğitim Seviyesi")
plt.show()

# %% slicing
x = data[['Egitim Seviyesi']]
y = data[['maas']]

# %% LinearRegression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)

plt.scatter(x,y)
plt.plot(x, lr.predict(x))
plt.show()

# %% Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree = 2)

x_poly = pr.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_poly,y)

plt.scatter(x,y)
plt.plot(x, lr2.predict(pr.fit_transform(x)))
plt.show()

# %% degree = 4
pr = PolynomialFeatures(degree = 4)

x_poly = pr.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_poly,y)

plt.scatter(x,y)
plt.plot(x, lr2.predict(pr.fit_transform(x)))
plt.show()

# %% prediction

print(lr.predict([[11]]))
print(lr.predict([[6.6]]))

print(lr2.predict(pr.fit_transform([[6.6]])))
print(lr2.predict(pr.fit_transform([[11]])))
