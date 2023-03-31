from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from tensorflow import keras

data = keras.datasets.cifar10.load_data()

'''
data = make_moons(n_samples=10000, shuffle=True, noise=0.4)

X, y = data[0], data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

forest = []

tree_clf = DecisionTreeClassifier(random_state=42)

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}

grid_search = GridSearchCV(tree_clf, params, verbose=1, cv=3)

grid_search.fit(X_train, y_train)

print(grid_search.best_estimator_)

for i in range(n_trees):

    tree = clone(grid_search.best_estimator_)

    forest.append(tree)

for i in range(n_trees):

    forest[i].fit(mini_sets[i][0], mini_sets[i][1])

test_set_predictions = []

for i in X_test:

    prediction_list = []

    for tree in forest:

        i = i.reshape(1, -1)

        prediction = tree.predict(i)

        prediction_list.append(prediction[0])

    mode = max(set(prediction_list), key=prediction_list.count)

    test_set_predictions.append(mode)

error_counter = 0

for i in range(len(test_set_predictions)):

    if test_set_predictions[i] != y_test[i]:

        error_counter += 1

print(1 - (error_counter / len(test_set_predictions)))

'''