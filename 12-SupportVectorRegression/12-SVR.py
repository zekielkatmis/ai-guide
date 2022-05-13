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

#%% Scaling
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
sc2 = StandardScaler()

x_scaled = sc1.fit_transform(x)
y_scaled = sc2.fit_transform(y)

# %% SVR
from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_scaled, y_scaled)

"""
kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
değerlerini alabilir.
Bu değer çizilen çizginin eğiminin belirler(net bi açıklama değil)
"""

plt.scatter(x_scaled, y_scaled)
plt.plot(x_scaled, svr_reg.predict(x_scaled))

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))
