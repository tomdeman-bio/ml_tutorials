#! /usr/bin/env py

from __future__ import print_function
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


iris = load_iris()

X = iris.data
print ("IRIS X = \n")
print (X)
print ("\n\n")

y = iris.target
print ("IRIS labels, y = \n")
print (y)


# Plot the PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


# plot the first three PCA dimensions
# code is adapted taken from: 
# http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,cmap=plt.cm.rainbow, s=60, alpha=1,  edgecolor='black')
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

#plt.show()
fig.savefig('IRIS_PCA.pdf')


