import itertools
import matplotlib.pyplot as plt
import numpy as np
import warnings
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=1.60)
class_names = [ 'false', 'true' ]

print('X')
print(X)
pause()
print('y')
print(y)
pause()

from sklearn.model_selection import train_test_split
# Note, prior to scikit-learn v0.18, this should be:
# from sklearn.cross_validation import train_test_split

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50)
# and testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, alpha=0.6)

plt.savefig('blob.png', bbox_inches='tight')
print('plotted blob.png')
pause()

from sklearn.metrics import confusion_matrix

# "Support Vector Classifier"
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
print(clf)
pause()

# To better visualize what's happening here, let's create a quick convenience
# function that will plot SVM decision boundaries for us:
def plot_svc_decision_function(clf):
    """Plot the decision function for a 2D SVC"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
        y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
        Y, X = np.meshgrid(y, x)
        P = np.zeros_like(X)
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                P[i, j] = clf.decision_function([xi, yj])
        return plt.contour(X, Y, P, colors='k',
                           levels=[-1, 0, 1],
                           linestyles=['--', '-', '--'])

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, alpha=0.6)
plot_svc_decision_function(clf)
plt.savefig('svm_01.png', bbox_inches='tight')
print('plotted svm_01.png')
pause()

# From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
y_pred = clf.fit(X_train, y_train).predict(X_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('cm.png', bbox_inches='tight')
print('plotted cm.png')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('cm-norm.png', bbox_inches='tight')
print('plotted cm-norm.png')
pause()

