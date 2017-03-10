#! /usr/bin/env py
# Code is adapted from:
# http://scikit-learn.org/stable/modules/model_persistence.html
# http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier)
from sklearn.cross_validation import train_test_split
import pickle

n_estimators = 10
iris = load_iris()

X1 = iris.data
y1 = iris.target

# Shuffle the iris data and labels
idx = np.arange(X1.shape[0])
np.random.shuffle(idx)
X1 = X1[idx]
y1 = y1[idx]
 
# Keep the first 10 for classifying 
ExampleX = X1[:10]
Exampley = y1[:10]

# Keep the last 140 for training and testing
X = X1[-140:]
y = y1[-140:]

# Get training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5, random_state=0)

#Generate the RF model
RF = RandomForestClassifier(n_estimators=n_estimators)
RF = RF.fit(X_train, y_train)
RFscore = RF.score(X_test, y_test)                           
print ("Random Forest score:", RFscore)

# Save RF classifier to disk
filename = "Saved_RF.pkl"
pickle.dump(RF, open(filename, 'wb'))  #wb means "write binary"

# load the model from disk
loaded_RF = pickle.load(open(filename, 'rb')) #rb means "read binary"

# Make predictions for all of the held out set 
predictions = loaded_RF.predict(ExampleX)
print ("Actual:")
print (Exampley)

print ("Classified As:")
print (predictions)


