#! usr/bin/env python
#adapted from: http://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
# You must install XGBoost separately
# conda install -c aterrel xgboost=0.4.0

from __future__ import print_function
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score

n_estimators = 10
iris = load_iris()


X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print (accuracy)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))


