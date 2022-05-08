# %% import
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% data import
data = pd.read_csv("veriler.csv")

height = data[["boy"]]

data.head(5)

# %% class


class insan:
    boy = 180

    def kosmak(self, b):
        return b+10


ali = insan()

print(ali.boy)
print(ali.kosmak(9))

# %% missing value

data2 = pd.read_csv("eksikveriler.csv")

data2.isnull().values.sum()

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

age = data2.iloc[:, 1:4].values

imputer = imputer.fit(age[:, 1:4])
age[:, 1:4] = imputer.transform(age[:, 1:4])

# %% kategorik verileri encode etme

country = data.iloc[:, 0:1].values
print(country)


le = preprocessing.LabelEncoder()

country[:, 0] = le.fit_transform(data.iloc[:, 0])

ohe = preprocessing.OneHotEncoder()

country = ohe.fit_transform(country).toarray()
print(country)

# %% concatenation data
result = pd.DataFrame(data=country, index=range(22),
                      columns=['fr', 'tr', 'us'])

result2 = pd.DataFrame(data=age, index=range(22),
                       columns=['height', 'weight', 'age'])

sex = data2.iloc[:, -1].values

result3 = pd.DataFrame(data=sex, index=range(22), columns=["cinsiyet"])

data = pd.concat([result, result2], axis=1)

data = pd.concat([data, result3], axis=1)

# %% split data test and train
