import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Wine.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.decomposition import KernelPCA
k_pca = KernelPCA(n_components=2, kernel="rbf")
X_train = k_pca.fit_transform(X_train)
X_test = k_pca.transform(X_test)


