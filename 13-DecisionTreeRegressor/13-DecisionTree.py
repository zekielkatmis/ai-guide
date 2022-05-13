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

# %% Decision Tree
from sklearn.tree import DecisionTreeRegressor

DTR = DecisionTreeRegressor(random_state = 0)
DTR.fit(x,y)

plt.scatter(x, y, color="red")
plt.plot(x, DTR.predict(x), color="blue")
plt.show()

# %% Prediction
print(DTR.predict([[11]]))
print(DTR.predict([[6.6]]))

# %%
a = x + 0.5
b = x - 0.4

plt.scatter(x, y, color="red")
plt.plot(x, DTR.predict(x), color="blue")
plt.plot(x, DTR.predict(a), color="green")
plt.plot(x, DTR.predict(b), color="orange")

"""
farklı x değerleri de olsa tahmin çizgileri üst üste geldi
çünkü decision tree veriyi aralıklarla böldüğü için
ufak farklılıklardan oluşan değerler tahminleri etkilemiyor
bu yüzden değer 0.5te oynasa tahmin edilen maaş değeri aynı kalıyor
"""