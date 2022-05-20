import numpy as np
import pandas as pd

# %% data read
data = pd.read_csv('veriler.csv')

x = data.drop(["ulke", "cinsiyet"], axis=1) # data.iloc[:,1:4].values
y = data[["cinsiyet"]] # data.iloc[:,4:].values

# %% verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=0)

# %% Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test =  sc.transform(x_test)

# %% RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(criterion="entropy", n_estimators=10)
rfc.fit(X_train, y_train.values.ravel())

y_pred = rfc.predict(X_test)

# %% Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

with open("output.txt", "a") as f:
    print('Confusion matrix when n_estimators=5:', file=f)
    print(cm, file=f)
    print("", file=f)
# %% roc ve auc için probability
y_probability = rfc.predict_proba(X_test)

with open("output.txt", "a") as f:
    print('Probability', file=f)
    print(y_probability, file=f)
    print("", file=f)

from sklearn import metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, y_probability[:,0], pos_label='e')