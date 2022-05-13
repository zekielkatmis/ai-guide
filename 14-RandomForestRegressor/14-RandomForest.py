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
X = x.to_numpy()
Y = y.to_numpy()

# %% RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

#n_estimators => kaç adet decision tree çizileceğini belirtir
RFR = RandomForestRegressor(random_state=0, n_estimators=10)
RFR.fit(X,Y.ravel())

print(RFR.predict([[6.6]]))

"""
virgüllü x değerleri de olsa decision tree veriyi aralıklarla böldüğü için
ufak farklılıklardan oluşan değerler tahminleri etkilemiyor
bu yüzden değer 0.5te oynasa da tahmin edilen maaş değeri aynı kalıyordu
ancak random forest birden fazla decision tree'den oluştuğu için
decison tree'lerin sonuçlarının ortalamasını alıp
6.6 gibi bir değer için 10500 tahminini yapabiliyor
bu değer tek bir decision tree'de 10000 idi
"""

Z = X + 0.5
K = X - 0.5

plt.scatter(X,Y, color = "red")
plt.plot(X,RFR.predict(X), color = "blue")
plt.plot(X,RFR.predict(Z), color = "green")
plt.plot(X,RFR.predict(K), color = "orange")