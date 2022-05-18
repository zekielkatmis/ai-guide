# %% import

import pandas as pd
import numpy as np

# %% data import

data = pd.read_csv("veriler.csv")

height = data[["boy"]]

data.head(5)

# %% missing value

from sklearn.impute import SimpleImputer

data2 = pd.read_csv("eksikveriler.csv")

data2.isnull().values.sum()

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

age = data2.iloc[:, 1:4].values

imputer = imputer.fit(age[:, 1:4])
age[:, 1:4] = imputer.transform(age[:, 1:4])

# %% kategorik verileri encode etme

from sklearn import preprocessing

country = data.iloc[:, 0:1].values

le = preprocessing.LabelEncoder()

country[:, 0] = le.fit_transform(data.iloc[:, 0])

ohe = preprocessing.OneHotEncoder()

country = ohe.fit_transform(country).toarray()

# %% concatenation data

result = pd.DataFrame(data=country, index=range(22),
                      columns=['fr', 'tr', 'us'])

result2 = pd.DataFrame(data=age, index=range(22),
                       columns=['height', 'weight', 'age'])

sex = data2.iloc[:, -1].values

y = pd.DataFrame(data=sex, index=range(22), columns=["cinsiyet"])

x = pd.concat([result, result2], axis=1)

data = pd.concat([x, y], axis=1)

# %% split data test and train

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=0)

# %% Scaling birbiri için anlamlı hale getirme işlemi

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
