# k-NN regression - Cross-Validated Predictions

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

np.random.seed(0)
X = np.sort(5 * np.random.rand(60, 1), axis=0)
y = np.sin(X).ravel()

# Add noise to targets (note that the 12 below is 60/5).
y[::5] += 1 * (0.5 - np.random.rand(12))

print('train X')
print(X)
print('train y')
print(y)
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

# Get the prediction of each value when it is in the test set.
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(knn, X, y, cv=10)

plt.figure()
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.savefig('knn_02.png', bbox_inches='tight')
print('plotted knn_02.png')
pause()

###############################################################################
# Look at the learning rate

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.5, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Learning Curves (KNN)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(knn, title, X, y, (0.7, 1.01), cv=cv)
plt.savefig('knn_lr.png', bbox_inches='tight')
print('plotted knn_lr.png')
pause()

