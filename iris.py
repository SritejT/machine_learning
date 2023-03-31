import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from CustomFiles.SoftmaxClassifier import SoftmaxClassifier
from sklearn.preprocessing import StandardScaler

X = pd.read_csv('datasets/iris/datasets_17860_23404_IRIS.csv')

y = X['species'].values

del X['species']

print(X)

def one_hot(x):

    new_x = []

    for i in range(0, len(x)):

        if x[i] == "Iris-versicolor":

            new_x.append(1)
            new_x.append(0)
            new_x.append(0)

        elif x[i] == "Iris-setosa":

            new_x.append(0)
            new_x.append(1)
            new_x.append(0)

        else:

            new_x.append(0)
            new_x.append(0)
            new_x.append(1)

    return np.asarray(new_x).reshape(120, 3)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

y_train = np.asarray(one_hot(y_train))

sft_clf = SoftmaxClassifier(eta=0.01, iterations=5001, method='batch')

sft_clf.fit(X_train, y_train)

X_test = scaler.fit_transform(X_test)

predictions = sft_clf.predict(X_test)

for i in range(len(y_test)):

    if y_test[i] == "Iris-versicolor":

        y_test[i] = 0

    elif y_test[i] == "Iris-setosa":

        y_test[i] = 1

    else:

        y_test[i] = 2

error_counter = 0

for i in range(len(y_test)):

    if y_test[i] != predictions[i]:

        error_counter += 1

print(error_counter / len(y_test))







