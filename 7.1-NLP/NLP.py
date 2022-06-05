import numpy as np
import pandas as pd

# %% read data

data = pd.read_csv('Restaurant_Reviews.csv', on_bad_lines = "skip")
