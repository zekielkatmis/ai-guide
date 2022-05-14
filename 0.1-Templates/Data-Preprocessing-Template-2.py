# %% import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% data read
data = pd.read_csv('veriler.csv')

# %% encoding country
from sklearn import preprocessing

country = data.iloc[:, 0:1].values

le = preprocessing.LabelEncoder()
country[:, 0] = le.fit_transform(data.iloc[:, 0])

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()

# %% encoding sex
gender = data.iloc[:, -1:].values

# man = 0 - woman = 1
le = preprocessing.LabelEncoder()
gender[:, -1] = le.fit_transform(data.iloc[:, -1])

# dummy variable 
# ohe = preprocessing.OneHotEncoder()
# sex = ohe.fit_transform(sex).toarray()

# %% cleaned data
HWA = data.iloc[:,1:4].values # HeightWeightAge

result1 = pd.DataFrame(data=country, index=range(22),
                       columns=['fr', 'tr', 'us'])

result2 = pd.DataFrame(data=HWA, index=range(
    22), columns=['boy', 'kilo', 'yas'])

result3 = pd.DataFrame(data=gender, index=range(22), columns=['cinsiyet'])

# dummy variable 
# result3 = pd.DataFrame(data=sex[:, :1], index=range(22), columns=['cinsiyet'])

# %% concat
s = pd.concat([result1, result2], axis=1)

processedData = pd.concat([s, result3], axis=1)

# %% verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    s, result3, test_size=0.33, random_state=0)
