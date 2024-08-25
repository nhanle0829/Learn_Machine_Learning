import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values   # Feature variables data
y = dataset.iloc[:, -1].values   # Dependent variables data
print("Importing dataset")
print(f"{X}\n\n{y}\n")

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
print("Taking care of missing values")
print(f"{X}\n")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
print("Encoding independent variable")
print(f"{X}\n")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print("Encoding dependent variable")
print(f"{y}\n")