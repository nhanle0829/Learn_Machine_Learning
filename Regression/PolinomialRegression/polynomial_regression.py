import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

plt.scatter(X, y, color="blue")
plt.plot(X, lin_reg.predict(X), color="red")
plt.title("Linear Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
# plt.show()

plt.scatter(X, y, color="blue")
plt.plot(X, lin_reg2.predict(X_poly), color="red")
plt.title("Polynomial Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
# plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="blue")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color="red")
plt.title("Polynomial Regression Model - Smooth Curved")
plt.xlabel("Position Level")
plt.ylabel("Salary")
# plt.show()

print(lin_reg.predict([[6.5]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))
