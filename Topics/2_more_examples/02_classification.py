import matplotlib.pyplot as plt
import numpy as np
import warnings
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)

print('X')
print(X)
pause()
print('y')
print(y)
pause()

plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
plt.savefig('blob.png', bbox_inches='tight')

# "Support Vector Classifier"
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X, y)
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

plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
plot_svc_decision_function(clf)
plt.savefig('svm_01.png', bbox_inches='tight')
print('plotted svm_01.png')
pause()

#Notice that the dashed lines touch a couple of the points: these points are
#known as the "support vectors", and are stored in the support_vectors_
#attribute of the classifier:
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=200, facecolors='red')
plt.savefig('svm_02.png', bbox_inches='tight')
print('plotted svm_02.png')
pause()

#The above version uses a linear kernel; it is also possible to use radial basis
# function kernels as well as others.
clf = SVC(kernel='rbf')
clf.fit(X, y)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200, facecolors='none')
plt.savefig('svm_03.png', bbox_inches='tight')
print('plotted svm_03.png')
