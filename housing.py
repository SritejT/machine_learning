from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_csv('datasets/housing/housing.csv')

y = data['median_house_value']

del data['longitude'], data['latitude'], data['ocean_proximity']

median = data['total_bedrooms'].median()

data['total_bedrooms'].fillna(median, inplace=True)

X = data

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)


def build_model(n_neurons=30, n_hidden=2, learning_rate=0.001):

    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=[7]))

    for layer in range(n_hidden):

        model.add(keras.layers.Dense(n_neurons, activation="relu"))

    model.add(keras.layers.Dense(1))

    optimizer = keras.optimizers.SGD(lr=learning_rate)

    model.compile(loss="mse", optimizer=optimizer)

    return model


keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

keras_reg.fit(X_train_scaled, y_train, epochs=100,
              validation_data=(X_valid_scaled, y_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])

param_distribs = {
    "n_hidden": [0, 1, 2, 3, 4, 5],
    "n_neurons": np.arange(1, 100),
    "learning_rate": [1, 0.1, 0.01, 0.001, 0.0001]
}

rnd_search = RandomizedSearchCV(keras_reg, param_distribs, cv=3, n_iter=10)

rnd_search.fit(X_train_scaled, y_train, epochs=100,
               validation_data=(X_valid_scaled, y_valid),
               callbacks=[keras.callbacks.EarlyStopping(patience=10)])

print(rnd_search.best_params_)


income = int(input("Median Income:"))
total_rooms = int(input("Total rooms in neighbourhood:"))
households = int(input("Total households in neighbourhood:"))
total_bedrooms = int(input("Total bedrooms in neighbourhood:"))
population = int(input("Total population of neighbourhood:"))
age = int(input("Age of housing:"))

use_data = {"housing_median_age": age, "total_rooms": [total_rooms], "total_bedrooms": [total_bedrooms],
            "population": [population], "households": [households], "median_income": [income / 10000]}


use_data = pd.DataFrame(data=use_data, index=[1])




