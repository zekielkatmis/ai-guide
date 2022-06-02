import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
# %% data read
data = pd.read_csv("Ads_CTR_Optimisation.csv")

# %% Thompson sampling
N = 10000  # 10bin satır var
d = 10  # toplam 10 reklam var
awards = [0] * d  # tüm reklamların ödülü başta 0
clicked = [0] * d  # o ana kadarki tıklamalar
totalAward = 0  # toplam ödül
selected = []
ones = [0] * d
zeros = [0] * d

for n in range(1, N):
    ad = 0  # seçilen ilan
    max_th = 0

    for i in range(0, d):  # max'tan büyük bir ucb çıktı
        randBeta = random.betavariate(ones[i] + 1, zeros[i] + 1)
        if randBeta > max_th:
            max_th = randBeta
            ad = i
    selected.append(ad)   
    award = data.values[n, ad]
    
    if award == 1:
        ones[ad] = ones[ad] + 1
    else:
        zeros[ad] = zeros[ad] + 1
        
    totalAward = totalAward + award

print("Toplam ödül: ")
print(totalAward)

plt.hist(selected)
plt.show()
