import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

start_time = time.time()

X_train = pd.read_csv('datasets/mnist/train.csv')

X_test = pd.read_csv('datasets/mnist/test.csv')

y_train = X_train['label']

del X_train['label']

X_train = np.asarray(X_train.values)
y_train = np.asarray(y_train.values)


tsne = TSNE(n_components=2)
pca = PCA(n_components=0.95)

pipeline = Pipeline([
    ('pca', pca),
    ('tsne', tsne),
])

shuffle_idx = np.random.permutation(42000)[:10000]

X_train_reduced = pipeline.fit_transform(X_train[shuffle_idx])

plt.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c=y_train[shuffle_idx])
plt.colorbar()
plt.show()
