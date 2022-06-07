import numpy as np
import pandas as pd

# %% read data
data = pd.read_csv('reviews.csv')
data = data.drop("Unnamed: 0", axis=1)

# %% preprocessing
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download("stopwords") # anlamsız kelimeleri çıkarmak için
ps = PorterStemmer () # kelimelerin eklerini atıp köklerini elde etmek için

reviews = []

for i in range (716):
    # a-z veya A-Z olmayan karakterleri bul boşluk ile değiştir
    # ^ işareti "değil"i temsil eder
    comment = re.sub("[^a-zA-Z]"," ",data["Review"][i])
    comment = comment.lower() # tüm harfleri küçült
    comment = comment.split() # kelime kelime listele
    
    # kelime(word) stopwords'ler içinde yoksa bu kelimeyi stem'le
    # kelimeyi listenin bir elemanı yap(köşeli parantezler bunun için)
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words("english"))]
    comment = " ".join(comment) # liste içindeki kelimeleri aralarına boşluk koyarak birleştir
    reviews.append(comment)
    
# %% feature extraction
from sklearn.feature_extraction.text import CountVectorizer

CV = CountVectorizer(max_features= 2000) # en fazla kullanılan 1000 kelimeyi al
x = CV.fit_transform(reviews).toarray() # bağımsız değişken
y = data["Liked"].values.astype(np.int64) # bağımlı değişken

# %% Split data test and train
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=0)

# %% classification
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

# %% confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

with open("output.txt", "a") as f:
    print('Confusion Matrix:', file=f)
    print(cm, file=f)