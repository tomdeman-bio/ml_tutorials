"""
============================
Nearest Neighbors regression
============================

Demonstrate the resolution of a regression problem
using a k-Nearest Neighbor and the interpolation of the
target using both barycenter and constant weights.

"""
###############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
XX = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

print('train X')
print(X)
print('train y')
print(y)
print('test X')
print(XX)
pause()

plt.scatter(X, y, c='k', label='train')
plt.axis('tight')
plt.legend()
plt.savefig('knn_01.png', bbox_inches='tight')
print('plotted knn_01.png')
pause()

###############################################################################
# Fit regression model
n_neighbors = 5
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
regr = knn.fit(X, y)
print(knn)
pause()

yy_ = regr.predict(XX)
print('predicted values')
print(yy_)
pause()

plt.clf()
plt.scatter(X, y, c='k', label='data')
plt.plot(XX, yy_, c='g', label='prediction')
plt.axis('tight')
plt.legend()
plt.savefig('knn_02.png', bbox_inches='tight')
print('plotted knn_02.png')
pause()
