# %% import
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# %% data import
data = pd.read_csv("maaslar_yeni.csv")
data = data.drop("Calisan ID", axis=1)

x = data.drop("maas", axis=1)
x = x.drop("unvan", axis=1)
#x = data[["UnvanSeviyesi"]]
y = data[["maas"]]

"""
print(data.corr())
UnvanSeviyesi/maas ilişkisi => 0.727036
"""
# %% Multiple Linear Regression
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(x, y)

mlr_pred = mlr.predict(x)

with open("output.txt", "a") as f:
    print('Multiple Linear Regression R2 score:', file=f)
    print(r2_score(y, mlr_pred), file=f)
    print("", file=f)
"""
import statsmodels.api as sm

model = sm.OLS(mlr_pred, x)
print(model.fit().summary())

Yalnızca Unvan Seviyesi kullanmamız gerektiğini p değerlerine bakarak anladık.
"""
# %% Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree = 4)
x_poly = pr.fit_transform(x)
x_poly_test = pr.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_poly,y)

lr2_pred = lr2.predict(x_poly_test)

with open("output.txt", "a") as f:
    print('Polynomial Regression R2 score:', file=f)
    print(r2_score(y, lr2_pred), file=f)
    print("", file=f)

# %% Scaling
from sklearn.preprocessing import StandardScaler

y_np = y.to_numpy()
x_np = x.to_numpy()

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(x_np)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(y_np.reshape(-1,1)))

# %% Support vector regressor
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

svr_reg_pred = svr_reg.predict(x_olcekli)

with open("output.txt", "a") as f:
    print('Support vector regressor R2 score:', file=f)
    print(r2_score(y_olcekli, svr_reg_pred), file=f)
    print("", file=f)

# %% Decision Tree regressor
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y)

r_dt_pred = r_dt.predict(x)

with open("output.txt", "a") as f:
    print('Decision Tree R2 score:', file=f)
    print(r2_score(y, r_dt_pred), file=f)
    print("", file=f)

# %% Random Forest regressor
from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(x,y_np.ravel())

rf_reg_pred = rf_reg.predict(x)

with open("output.txt", "a") as f:
    print('Random Forest R2 score:', file=f)
    print(r2_score(y, rf_reg_pred), file=f)