import matplotlib.pyplot as plt
import numpy as np
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

np.random.seed(1)
X = np.dot(np.random.random(size=(2, 2)), np.random.normal(size=(2, 200))).T
plt.plot(X[:, 0], X[:, 1], 'og')
plt.axis('equal')
plt.savefig('pca_01.png', bbox_inches='tight')
print('plotted pca_01.png')
pause()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
print('pca.explained_variance_')
print(pca.explained_variance_)
print('pca.components_')
print(pca.components_)
pause()

plt.clf()
plt.plot(X[:, 0], X[:, 1], 'og', alpha=0.3)
plt.axis('equal')
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw=3)
plt.savefig('pca_02.png', bbox_inches='tight')
print('plotted pca_02.png')
pause()

clf = PCA(0.95)
X_trans = clf.fit_transform(X)
print('X.shape')
print(X.shape)
print('X_trans.shape')
print(X_trans.shape)
pause()

X_new = clf.inverse_transform(X_trans)
plt.clf()
plt.plot(X[:, 0], X[:, 1], 'og', alpha=0.2)
plt.plot(X_new[:, 0], X_new[:, 1], 'og', alpha=0.8)
plt.axis('equal')
plt.savefig('pca_03.png', bbox_inches='tight')
print('plotted pca_03.png')
pause()

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=0.60)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.savefig('kmeans_01.png', bbox_inches='tight')
print('plotted kmeans_01.png')
pause()

from sklearn.cluster import KMeans
est = KMeans(4)  # 4 clusters
est.fit(X)
y_kmeans = est.predict(X)
print('y_kmeans')
print(y_kmeans)
pause()

plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50)
plt.savefig('kmeans_02.png', bbox_inches='tight')
print('plotted kmeans_02.png')
pause()
