from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt
import numpy as np

X = np.load('datasets/olivetti/olivetti_faces.npy')
y = np.load('datasets/olivetti/olivetti_faces_target.npy')

train_test_shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in train_test_shuffle_split.split(X, y):

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

X_train = X_train.reshape(320, 4096)
X_test = X_test.reshape(80, 4096)

pca = PCA(n_components=0.99)
bgm = BayesianGaussianMixture()

bgm.fit(X_train)

new_faces, y_new_faces = bgm.sample(n_samples=20)

faces_list = []

for i in new_faces:

    face = i.reshape(64, 64).T

    faces_list.append(face)

bad_faces = np.asarray(faces_list).reshape(20, 4096)

def reconstruction_error(X):

    if X.all() == X_train.all():

        reduced_X = pca.fit_transform(X)

    else:

        reduced_X = pca.transform(X)

    reconstructed_X = pca.inverse_transform(reduced_X)

    mse = np.power(reconstructed_X - X, 2)

    return mse.mean()


print(reconstruction_error(X_train))

print(reconstruction_error(bad_faces))

















