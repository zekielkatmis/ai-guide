# %% kutuphaneler
import numpy as np
import pandas as pd

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

# %% verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
