import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = []
for i in range(7501):
    transactions.append([str(dataset.values[i, j]) for j in range(20)])

from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
