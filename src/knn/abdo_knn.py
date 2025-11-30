# src/knn/abdo_knn.py

from collections import Counter
from math import sqrt


class ABDOKNNClassifier:
    def __init__(self, k: int = 3):
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k

    def fit(self, X, y):
        if len(X) == 0:
            raise ValueError("X cannot be empty")

        if len(X) != len(y):
            raise ValueError("Number of elements in X and y must be equal")

        self._X_train = X
        self._y_train = y

    def _distance(self, p1, p2):
        powers = 0
        for x in range(len(p1)):
            powers += (p2[x]-p1[x]) ** 2
        distance = sqrt(powers)

        return distance

    def _predict_one(self, x):
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("First you must use fit(X, y)")

        distances = []
        for x_train, y_train in zip(self._X_train, self._y_train):
            d = self._distance(x, x_train)
            distances.append((d, y_train))

        distances.sort()
        k_nearest = distances[:self.k]

        labels = []
        for x in k_nearest:
            labels.append(x[1])

        counter = Counter(labels)
        max_label = max(counter, key=counter.get)
        return max_label

    def predict(self, X):
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("First you must use fit(X, y)")

        predictions = []
        for x in X:
            label = self._predict_one(x)
            predictions.append(label)
        return predictions