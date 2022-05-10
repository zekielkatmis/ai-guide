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

# %% encoding gender
gender = data.iloc[:, -1:].values

# man = 0 - woman = 1
le = preprocessing.LabelEncoder()
gender[:, -1] = le.fit_transform(data.iloc[:, -1])

# dummy variable 
# ohe = preprocessing.OneHotEncoder()
# gender = ohe.fit_transform(gender).toarray()

# %% cleaned data
HWA = data.iloc[:,1:4].values # HeightWeightAge

result1 = pd.DataFrame(data=country, index=range(22),
                       columns=['fr', 'tr', 'us'])

result2 = pd.DataFrame(data=HWA, index=range(
    22), columns=['boy', 'kilo', 'yas'])

result3 = pd.DataFrame(data=gender, index=range(22), columns=['cinsiyet'])

# dummy variable 
# result3 = pd.DataFrame(data=gender[:, :1], index=range(22), columns=['cinsiyet'])

# %% concat
s = pd.concat([result1, result2], axis=1)

processedData = pd.concat([s, result3], axis=1)

# %% verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    s, result3, test_size=0.33, random_state=0)

# %% Multiple Linear Regression (gender)
from sklearn.linear_model import LinearRegression

regressorGender = LinearRegression()
regressorGender.fit(x_train, y_train)

y_predGender = regressorGender.predict(x_test)

#from sklearn.metrics import mean_absolute_error
#genderError = mean_absolute_error(y_test, y_predGender)

# %% Multiple Linear Regression (height)
height = processedData["boy"].values
newData = processedData.drop("boy", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    newData, height, test_size=0.33, random_state=0)

regressorHeight = LinearRegression()
regressorHeight.fit(x_train, y_train)

y_predHeight = regressorHeight.predict(x_test)

#heightError = mean_absolute_error(y_test, y_predHeight)

# %% model başarısı
import statsmodels.api as sm

"""
 MLR formülü => y=b0+b1x1+b2x2+b3x3+bİxİ+ϵ
 
 MLR'da β₀ sabiti gerekmekte
 ancak bizim verimizde sabit bir değer bulunmuyor.
 Bu yüzden veri setimize sabit bir kolon eklemeliyiz.
 
 Değişken seçimi yapacağımız için bu işlemi yapmalıyız.
 
 Aşağıda p-değerlerini hesaplayacağım
"""

X = np.append(arr= np.ones((22,1)).astype(int), 
              values=newData, axis=1)

X_list = newData.iloc[:,[0,1,2,3,4,5]].values
X_list = np.array(X_list, dtype=float)
model = sm.OLS(height, X_list).fit()

with open("output.txt", "a") as f:
    print(model.summary(), file=f)
    
"""
ilk outputta x5 yani 4. kolondaki p-değeri
0.05 olarak belirlediğimiz significance değerinden büyük
bu yüzden sıradaki modelde 4. kolonu çıkarıyoruz
"""

X_list = newData.iloc[:,[0,1,2,3,5]].values
X_list = np.array(X_list, dtype=float)
model = sm.OLS(height, X_list).fit()

with open("output.txt", "a") as f:
    print(model.summary(), file=f)
    
"""
ikinci outputta x5 yani yin 4. kolondaki p-değeri 0.031
0.05 olarak belirlediğimiz significance değerine yakın
sıradaki modelde 4. kolonu da çıkarabiliriz (zorunlu değil)
"""

X_list = newData.iloc[:,[0,1,2,3]].values
X_list = np.array(X_list, dtype=float)
model = sm.OLS(height, X_list).fit()

with open("output.txt", "a") as f:
    print(model.summary(), file=f)
    
"""
son outputta tüm p değerlerini 0 olarak görüyoruz
"""