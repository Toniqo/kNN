#!/usr/bin/env python3

import time

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from knn.abdo_knn import ABDOKNNClassifier


def main():
    data = load_digits()
    # data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Our implementation wants python lists instead of numpy array
    X_train_list = X_train.tolist()
    X_test_list = X_test.tolist()
    y_train_list = y_train.tolist()
    y_test_list = y_test.tolist()

    k = int(input("Provide number of neighbors: "))

    start = time.perf_counter()
    abdo_knn = ABDOKNNClassifier(k=k)
    abdo_knn.fit(X_train_list, y_train_list)
    y_pred_abdo = abdo_knn.predict(X_test_list)
    acc_abdo = accuracy_score(y_test_list, y_pred_abdo)
    end = time.perf_counter()
    time_abdo = end - start

    start = time.perf_counter()
    skl_knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    skl_knn.fit(X_train, y_train)
    y_pred_skl = skl_knn.predict(X_test)
    acc_skl = accuracy_score(y_test, y_pred_skl)
    end = time.perf_counter()
    time_skl = end - start

    print(f"ABDO k-NN accuracy: {acc_abdo:.3f}")
    print(f"Scikit-learn k-NN accuracy: {acc_skl:.3f}")
    print(f"ABDO program time: {time_abdo:.5f}")
    print(f"Scikit-learn program time: {time_skl:.5f}")

if __name__ == "__main__":
    main()
