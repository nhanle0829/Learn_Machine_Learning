import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values   # Feature variables data
Y = dataset.iloc[:, -1].values   # Dependent variables data
print(X)
print(Y)