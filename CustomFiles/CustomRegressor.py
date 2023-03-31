import numpy as np

np.random.seed(42)

class CustomRegressor:

    def __init__(self, regularisation, iterations=0, eta=0):

        self.regularisation = regularisation
        self.iterations = iterations
        self.eta = eta

    def fit(self, X, y):

        X = np.array(np.c_[np.ones((X.shape[0], 1)), X])

        y = y.values

        print(X.shape[1])

        if self.regularisation == "none":

            theta = np.random.randn(X.shape[1], 1)

            for iteration in range(self.iterations):

                idx = np.random.randint(X.shape[0])

                X_instance = X[idx]

                y_instance = y[idx]

                logit = X_instance.dot(theta)

                cost = ((logit - y_instance) ** 2) ** (1/2)

                gradient = (2 / X_instance.shape[0]) * X_instance.T * (logit - y_instance)[0]

                print(gradient.shape)

                gradient = np.asarray(gradient).reshape(18, 1)

                theta -= self.eta * gradient

        elif self.regularisation[0] == "ridge":

            alpha = self.regularisation[1]

            X_squared = X.T.dot(X)

            A = np.identity(X_squared.shape[0])

            A[0][0] = 0

            theta = np.linalg.inv(X_squared + (alpha * A)).dot(X.T).dot(y)

        else:

            raise ValueError("Regularisation is not defined properly (none or ridge)")

        self.theta = theta

    def predict(self, X):

        prediction_list = []

        X = np.array(np.c_[np.ones((X.shape[0], 1)), X])

        for i in X:

            prediction = i.dot(self.theta)

            prediction_list.append(prediction)

        return prediction_list
