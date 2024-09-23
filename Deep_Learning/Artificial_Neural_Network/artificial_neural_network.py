import numpy as np
import pandas as pd
import tensorflow as tf
import keras

print(tf.__version__)

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = keras.models.Sequential()

ann.add(keras.layers.Dense(units=6, activation="relu"))
ann.add(keras.layers.Dense(units=6, activation="relu"))

ann.add(keras.layers.Dense(units=1, activation="sigmoid"))

ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
ann.fit(X_train, y_train, batch_size=32, epochs=100)

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]) > 0.5))

y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5
print(np.concatenate((y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1))

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
