# %% import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% data read
data = pd.read_csv('odev_tenis.csv')

# %% encoding country
from sklearn import preprocessing

outlook = data["outlook"].values

le = preprocessing.LabelEncoder()
outlook = le.fit_transform(data["outlook"])

ohe = preprocessing.OneHotEncoder()
outlook = outlook.reshape(-1,1)
outlook = ohe.fit_transform(outlook).toarray()

# %% encoding windy-play
windy = data["windy"].values
play = data["play"].values

# false = 0 - true = 1
windy = le.fit_transform(windy)
# no = 0 - yes = 1
play = le.fit_transform(play)

# %% cleaned data
tempHum = data.iloc[:,1:3].values # temperature-humidity

r1 = pd.DataFrame(data=outlook, index=range(14), columns=['overcast', 'rainy', 'sunny'])
r2 = pd.DataFrame(data=tempHum, index=range(14), columns=['temperature', 'humidity'])
r3 = pd.DataFrame(data=windy, index=range(14), columns=['windy'])
r4 = pd.DataFrame(data=play, index=range(14), columns=['play'])

# %% concat
processedData = pd.concat([r1, r2, r3, r4], axis=1)

y = processedData["play"].values
y = pd.DataFrame(y)
x = processedData.drop("play", axis=1)

# %% verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=0)

# %% Multiple Linear Regression (play)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# %% model başarısı
import statsmodels.api as sm

X = np.append(arr= np.ones((14,1)).astype(int), 
              values=x, axis=1)

X_list = x.iloc[:,[0,1,2,3,4,5]].values
X_list = np.array(X_list, dtype=float)
model = sm.OLS(y, X_list).fit()

with open("output.txt", "a") as f:
    print(model.summary(), file=f)
"""
X_list = x.iloc[:,[0,1,2,4,5]].values
X_list = np.array(X_list, dtype=float)
model = sm.OLS(y, X_list).fit()

with open("output.txt", "a") as f:
    print(model.summary(), file=f)
    
X_list = x.iloc[:,[0,1,2,5]].values
X_list = np.array(X_list, dtype=float)
model = sm.OLS(y, X_list).fit()

with open("output.txt", "a") as f:
    print(model.summary(), file=f)
    
X_list = x.iloc[:,[0,1,2]].values
X_list = np.array(X_list, dtype=float)
model = sm.OLS(y, X_list).fit()

with open("output.txt", "a") as f:
    print(model.summary(), file=f)
"""
# %%
x_train = x_train.drop("temperature", axis=1)
x_test = x_test.drop("temperature", axis=1)
regressor.fit(x_train, y_train)
y_pred2 = regressor.predict(x_test)

"""
x_train = x_train.drop("humidity", axis=1)
x_test = x_test.drop("humidity", axis=1)
regressor.fit(x_train, y_train)
y_pred3 = regressor.predict(x_test)

x_train = x_train.drop("windy", axis=1)
x_test = x_test.drop("windy", axis=1)
regressor.fit(x_train, y_train)
y_pred4 = regressor.predict(x_test)
"""