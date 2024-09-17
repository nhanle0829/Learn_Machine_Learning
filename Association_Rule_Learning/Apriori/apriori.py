import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = []
for i in range(7501):
    transactions.append([str(dataset.values[i, j])for j in range(20)])
