#!/usr/bin/env python3

from knn.abdo_knn import ABDOKNNClassifier

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt


def main():
    data = load_digits()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    X_train_list = X_train.tolist()
    X_test_list = X_test.tolist()
    y_train_list = y_train.tolist()
    y_test_list = y_test.tolist()

    acc_abdo = []
    acc_skl = []

    # k = 1-16
    ks = list(range(1, 16))

    for k in ks:
        abdo_knn = ABDOKNNClassifier(k=k)
        abdo_knn.fit(X_train_list, y_train_list)
        y_pred_abdo = abdo_knn.predict(X_test_list)
        acc_abdo.append(accuracy_score(y_test_list, y_pred_abdo))

        skl = KNeighborsClassifier(n_neighbors=k)
        skl.fit(X_train, y_train)
        y_pred_skl = skl.predict(X_test)
        acc_skl.append(accuracy_score(y_test, y_pred_skl))

    plt.figure()
    plt.plot(ks, acc_skl, marker='s', label="KNeighborsClassifier (scikit-learn)")
    plt.plot(ks, acc_abdo, marker='o', label="ABDO kNN")
    plt.xlabel("k (neighbors)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over k")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    k_plt = 5
    abdo = ABDOKNNClassifier(k=k_plt)
    abdo.fit(X_train, y_train)
    y_pred_abdo = abdo.predict(X_test)

    skl = KNeighborsClassifier(n_neighbors=k_plt)
    skl.fit(X_train, y_train)
    y_pred_skl = skl.predict(X_test)

    cm_abdo = confusion_matrix(y_test, y_pred_abdo)
    cm_skl = confusion_matrix(y_test, y_pred_skl)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    im0 = axes[0].imshow(cm_abdo)
    axes[0].set_title(f"ABDO kNN, k={k_plt}")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")

    im1 = axes[1].imshow(cm_skl)
    axes[1].set_title(f"scikit-learn KNN, k={k_plt}")
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")

    fig.colorbar(im0, ax=axes.ravel())
    fig.suptitle("Confusion matrix: ABDO kNN vs scikit-learn kNN")
    plt.show()

if __name__ == "__main__":
    main()
