# %% Import
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# %% Import data
data = pd.read_csv('maaslar.csv')

# %% Slicing
x = data[['Egitim Seviyesi']]
y = data[['maas']]
X = x.to_numpy()
Y = y.to_numpy()

# %% linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

with open("output.txt", "a") as f:
    print('Linear R2 score:', file=f)
    print(r2_score(y, lr.predict(X)), file=f)
    print("", file=f)

# %% Polynomial regression
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree = 4)
x_poly = pr.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_poly,y)

with open("output.txt", "a") as f:
    print('Polynomial R2 score:', file=f)
    print(r2_score(y, lr2.predict(pr.fit_transform(x))), file=f)
    print("", file=f)

# %% Scaling
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

# %% Support vector regressor
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

with open("output.txt", "a") as f:
    print('SVR R2 score:', file=f)
    print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)), file=f)
    print("", file=f)

# %% Decision Tree regressor
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y)

Z = x + 0.5
K = x - 0.4

with open("output.txt", "a") as f:
    print('Decision Tree R2 score:', file=f)
    print(r2_score(Y, r_dt.predict(X)), file=f)
    print("", file=f)

# %% Random Forest regressor
from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(x,Y.ravel())

with open("output.txt", "a") as f:
    print('Random Forest R2 score:', file=f)
    print(r2_score(Y, rf_reg.predict(X)), file=f)
    print("", file=f)
    print('Random Forest R2 score x-0.4 values:', file=f)
    print(r2_score(Y, rf_reg.predict(K)), file=f)
    print("", file=f)
    print('Random Forest R2 score x+0.5 values:', file=f)
    print(r2_score(Y, rf_reg.predict(Z)), file=f)