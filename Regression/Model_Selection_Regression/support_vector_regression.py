import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv") # Replace "Data.csv" with your dataset name
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(X_train, y_train)

y_pred = regressor.predict(sc_X.transform(X_test)).reshape(-1, 1)
np.set_printoptions(precision=2)
print(np.concatenate((y_test, sc_y.inverse_transform(y_pred)), axis=1))

from sklearn.metrics import r2_score
print(r2_score(y_test, sc_y.inverse_transform(y_pred)))
