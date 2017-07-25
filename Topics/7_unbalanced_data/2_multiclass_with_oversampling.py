import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


from imblearn.over_sampling import SMOTE

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import graph as g

RANDOM_STATE = 42

if (len(sys.argv) != 3):
	print 'requires arg1=X_fname and arg2=y_fname'
	sys.exit(1)
			 
X_fname = sys.argv[1] 
y_fname = sys.argv[2]

# Load the data from tab delimited files
X = np.loadtxt(open(X_fname, "rb"), dtype=float, delimiter='\t')
y = np.loadtxt(open(y_fname, "rb"), dtype=int, delimiter='\t')
class_names = np.unique(y)

print "X ", X.shape
print "y ", y.shape
print "class_names ", class_names

# Create the training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,
		test_size=0.33, random_state=RANDOM_STATE)

print "X_train ", X_train.shape
print "y_train ", y_train.shape



# Create and fit a model using linear support vector classifier
svc = LinearSVC(random_state=RANDOM_STATE)
svc.fit(X_train, y_train)

# Classify test data and report the results
print(classification_report(y_test, svc.predict(X_test)))
cnf_matrix = cnf_matrix = confusion_matrix(y_test, svc.predict(X_test))

# Create a confusion matrix
np.set_printoptions(precision=2)
plt.figure()
g.plot_confusion_matrix(cnf_matrix, classes=class_names,
	title='Confusion matrix, without Oversampling')
plt.draw()

# Oversample the data using SMOTE regular
sm = SMOTE(kind='regular')
X_train_resampled, y_train_resampled = sm.fit_sample(X_train, y_train)

print "X_train_resampled ", X_train_resampled.shape
print "y_train_resampled ", y_train_resampled.shape

# Create and fit a model using linear support vector classifier
svc = LinearSVC(random_state=RANDOM_STATE)
svc.fit(X_train_resampled, y_train_resampled)

print(classification_report(y_test, svc.predict(X_test)))
cnf_matrix = cnf_matrix = confusion_matrix(y_test, svc.predict(X_test))

# Create a confusion matrix
np.set_printoptions(precision=2)
plt.figure()
g.plot_confusion_matrix(cnf_matrix, classes=class_names,
		        title='Confusion matrix, SMOTE regular oversampling')
plt.draw()

# Oversample using SMOTE SVM
sm = SMOTE(kind='svm')
X_train_resampled, y_train_resampled = sm.fit_sample(X_train, y_train)

print "X_train_resampled ", X_train_resampled.shape
print "y_train_resampled ", y_train_resampled.shape

# Create and fit a model using linear support vector classifier
svc = LinearSVC(random_state=RANDOM_STATE)
svc.fit(X_train_resampled, y_train_resampled)

print(classification_report(y_test, svc.predict(X_test)))
cnf_matrix = cnf_matrix = confusion_matrix(y_test, svc.predict(X_test))

# Create a confusion matrix
np.set_printoptions(precision=2)
plt.figure()
g.plot_confusion_matrix(cnf_matrix, classes=class_names,
		title='Confusion matrix, SMOTE SVM Oversampling')
plt.draw()

plt.show()
