import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

print(regressor.predict([[6.5]]))

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(-1, 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Decision Tree Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()