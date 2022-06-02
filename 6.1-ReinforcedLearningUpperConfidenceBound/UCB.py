import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %% data read
data = pd.read_csv("Ads_CTR_Optimisation.csv")

# %% random selection
# random selection için random generator'a ihtiyacımız var
import random

N = 10000 # 10bin satır var
d = 10
total = 0
selected = []

for n in range(0,N):
    ads = random.randrange(d)
    selected.append(ads)
    award = data.values[n, ads] 
    # verideki n. satır=1 ise award=1
    total = total + award
"""
her satır için bir değer seçiyor
seçilen değer o satırda 1 ise total 1 artıyor
random olarak reklam atadığımızda kullanıcıların reklam tıklaması
1200 civarlarında olacakmış diyebiliyoruz
"""

plt.hist(selected)
plt.show()

# %% Upper Confidence Bound
# reklam tıklanmasını simüle ediyoruz
import math

N = 10000 # 10bin satır var
d = 10
awards = [0] * d # tüm reklamların ödülü başta 0
clicked = [0] * d # o ana kadarki tıklamalar
totalAward = 0 # toplam ödül
selected2 = []

for n in range(1,N):
    ad = 0 # seçilen ilan
    max_ucb = 0
    
    for i in range(0,d): # max'tan büyük bir ucb çıktı
        if(clicked[i] > 0):
            avg = awards[i] / clicked[i]
            delta = math.sqrt(3/2 * math.log(n)/clicked[i])
            ucb = avg + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
            
    selected2.append(ad)
    clicked[ad] = clicked[ad] + 1
    award2 = data.values[n, ad]
    awards[ad] = awards[ad] + award2
    totalAward = totalAward + award2

print("Toplam ödül: ")
print(totalAward)

plt.hist(selected2)
plt.show()