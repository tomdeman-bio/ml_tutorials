#! /usr/bin/env py
# Code is adapted from:
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

n_estimators = 10
iris = load_iris()


X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

DT = DecisionTreeClassifier(max_depth=3)
DT = DT.fit(X_train, y_train)
DTscore = DT.score(X_test, y_test)                           
DTimportance = DT.feature_importances_

RF = RandomForestClassifier(n_estimators=n_estimators)
RF = RF.fit(X_train, y_train)
RFscore = RF.score(X_test, y_test)                           
RFimportance = RF.feature_importances_

ET = ExtraTreesClassifier(n_estimators=n_estimators)
ET = ET.fit(X_train, y_train)
ETscore = ET.score(X_test, y_test)                           
ETimportance = ET.feature_importances_

AB = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=n_estimators)
AB = AB.fit(X_train, y_train)
ABscore = AB.score(X_test, y_test)                           
ABimportance = AB.feature_importances_

GB = GradientBoostingClassifier(n_estimators=n_estimators)
GB = GB.fit(X_train, y_train)
GBscore = GB.score(X_test, y_test)                           
GBimportance = GB.feature_importances_


print ("Decision Tree:", DTscore)
print ("Random Forest:", RFscore)
print ("Extra Trees:", ETscore)
print ("AdaBoost:", ABscore)
print ("Gradient Boost:", GBscore)
print ()

print ("Decsision Tree Importance")
# creating an array called indices that is the feature ssorted in descending order
indices = np.argsort(DTimportance)[::-1]
# this is saying for each element equal to the number of columns in the matrix X:
for f in range(X.shape[1]):   
    print("%d. feature %d (%f)" % (f + 1, indices[f], DTimportance[indices[f]]))
print()

print ("Random Forest Importance")
indices = np.argsort(RFimportance)[::-1]
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], RFimportance[indices[f]]))
print ()

print ("Extra Trees Importance")
indices = np.argsort(ETimportance)[::-1]
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], ETimportance[indices[f]]))
print ()

print ("AdaBoost Importance")
indices = np.argsort(ABimportance)[::-1]
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], ABimportance[indices[f]]))
print ()

print ("Gradient Boost Importance")
indices = np.argsort(GBimportance)[::-1]
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], GBimportance[indices[f]]))
print ()






































