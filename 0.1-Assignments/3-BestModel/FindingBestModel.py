import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
# %% data read
data = pd.read_excel('Iris.xls')

x = data.drop(["iris"], axis=1) # data.iloc[:,1:4].values
y = data[["iris"]]
# %% visualization
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d # noqa: F401
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
x2 = iris.data[:, :2]  # we only take the first two features.
y2 = iris.target

x_min, x_max = x2[:, 0].min() - 0.5, x2[:, 0].max() + 0.5
y_min, y_max = x2[:, 1].min() - 0.5, x2[:, 1].max() + 0.5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(x2[:, 0], x2[:, 1], c=y2, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y2,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()

# %% verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=0)

# %% Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test =  sc.transform(x_test)

# %% Logistic Regression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train.values.ravel())
y_pred = logr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
with open("output.txt", "a") as f:
    print('Confusion matrix for Logistic Regression:', file=f)
    print(cm, file=f)
    print("", file=f)
    
# %% KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
knn.fit(X_train, y_train.values.ravel())
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
with open("output.txt", "a") as f:
    print('Confusion matrix for KNeighborsClassifier:', file=f)
    print(cm, file=f)
    print("", file=f)
    
# %% SVM
from sklearn.svm import SVC

svc = SVC(kernel="rbf")
svc.fit(X_train, y_train.values.ravel())
y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
with open("output.txt", "a") as f:
    print('Confusion matrix for SVM:', file=f)
    print(cm, file=f)
    print("", file=f)
    
# %% Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train.values.ravel())
y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
with open("output.txt", "a") as f:
    print('Confusion matrix for Gaussian Naive Bayes:', file=f)
    print(cm, file=f)
    print("", file=f)

# %% DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="gini")
dtc.fit(X_train, y_train.values.ravel())
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
with open("output.txt", "a") as f:
    print('Confusion matrix for DecisionTreeClassifier:', file=f)
    print(cm, file=f)
    print("", file=f)

# %% RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(criterion="gini", n_estimators=50)
rfc.fit(X_train, y_train.values.ravel())
y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
with open("output.txt", "a") as f:
    print('Confusion matrix for RandomForestClassifier:', file=f)
    print(cm, file=f)
    print("", file=f)
    
# %% roc ve auc için probability
"""
from sklearn import metrics

y_probability = dtc.predict_proba(X_test)
with open("output.txt", "a") as f:
    print('Probability', file=f)
    print(y_probability, file=f)
    print("", file=f)

fpr, tpr, threshold = metrics.roc_curve(y_test, y_probability[:,0], pos_label='e')
"""
