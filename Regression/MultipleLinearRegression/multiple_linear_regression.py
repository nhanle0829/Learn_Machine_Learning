import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

