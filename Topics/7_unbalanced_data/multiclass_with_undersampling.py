import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import graphs as g

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

print X.shape
print y.shape
print class_names

# Create the training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,
		test_size=0.33, random_state=RANDOM_STATE)

print X_train.shape
print y_train.shape

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
	title='Confusion matrix, without normalization')
plt.show()

# Create a pipeline
pipeline = make_pipeline(NearMiss(version=2, random_state=RANDOM_STATE),
                        LinearSVC(random_state=RANDOM_STATE))
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
cnf_matrix = cnf_matrix = confusion_matrix(y_test, pipeline.predict(X_test))

# Create a confusion matrix
np.set_printoptions(precision=2)
plt.figure()
g.plot_confusion_matrix(cnf_matrix, classes=class_names,
		        title='Confusion matrix, without normalization')
plt.show()

