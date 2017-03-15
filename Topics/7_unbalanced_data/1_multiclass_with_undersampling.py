import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler

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
	title='Confusion matrix, without Undersampling')
plt.show()

# Undersample the data using NearMiss
nm2 = NearMiss(version=2)
X_train_resampled, y_train_resampled = nm2.fit_sample(X_train, y_train)

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
		        title='Confusion matrix, NearMiss2 undersampling')
plt.show()

# Undersample the data using TomekLinks
tl = TomekLinks()
X_train_resampled, y_train_resampled = tl.fit_sample(X_train, y_train)

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
		title='Confusion matrix, TomekLinks undersampling')
plt.show()

# Undersample the data using RandomUndersampler
rus = RandomUnderSampler()
X_train_resampled, y_train_resampled = rus.fit_sample(X_train, y_train)

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
		title='Confusion matrix, Random undersampling')
plt.show()


