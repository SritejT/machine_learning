import numpy as np

np.random.seed(2000)


def softmax(x):

    exp = np.exp(x)

    sum_exps = np.sum(exp, keepdims=True)

    return exp / sum_exps


class SoftmaxClassifier:

    def __init__(self, eta=0.01, iterations=5001, method='batch'):

        self.eta = eta
        self.iterations = iterations
        self.method = method

    def fit(self, X, y_train):

        print(X)

        X = np.asarray(np.c_[np.ones([X.shape[0], 1]), X])

        n_inputs = X.shape[1]

        n_outputs = len(np.unique(y_train))

        theta = np.random.randn(n_inputs, n_outputs)

        for iteration in range(self.iterations):

            if self.method == 'batch':

                X_instance = X

                y_instance = y_train

            elif self.method == 'mini-batch':

                idx = np.random.choice(len(X), size=round(len(X) / 10), replace=False)

                X_instance = X[idx]

                y_instance = y_train[idx]

            else:

                exit()

            logit = X_instance.dot(theta)

            proba = softmax(logit)

            error = proba - y_instance

            gradient = (1 / len(X_instance) * X_instance.T.dot(error))

            theta -= self.eta * gradient

        self.theta = theta

    def predict(self, X_test):

        predictions = []

        X_test = np.asarray(np.c_[np.ones([len(X_test), 1]), X_test])

        for i in X_test:

            softmax_score = i.T.dot(self.theta)

            logit = softmax(softmax_score)

            k = np.argmax(logit)

            predictions.append(k)

        return np.asarray(predictions)
