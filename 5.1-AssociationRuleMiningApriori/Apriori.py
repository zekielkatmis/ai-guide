import numpy as np
import pandas as pd

# %% import data
data = pd.read_csv("sepet.csv", header=None)

t = []
for i in range(0, 7501):
    t.append([str(data.values[i,j])
              for j in range (0,20)])

# %% apriori
from apyori import apriori

"""
min_support -- The minimum support of relations (float). 
min_confidence -- The minimum confidence of relations (float). 
min_lift -- The minimum lift of relations (float). 
max_length -- The maximum length of the relation (integer).
"""
rules = apriori(t, min_support=0.01, 
        min_confidence=0.2, 
        min_lift=3,
        min_length=2)

with open("output.txt", "a") as f:
    print('Rule list:', file=f)
    print(list(rules), file=f)
