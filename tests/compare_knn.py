# src/knn/compare_knn.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from knn.abdo_knn import ABDO_KNN_Classifier


def main():
    # 1. Ładujemy klasyczny zbiór Iris
    iris = load_iris()
    X = iris.data        # numpy array (n_samples, n_features)
    y = iris.target      # numpy array (n_samples,)

    # 2. Dzielimy na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. Konwersja do czystych list dla naszego prostego KNN
    X_train_list = X_train.tolist()
    X_test_list = X_test.tolist()
    y_train_list = y_train.tolist()
    y_test_list = y_test.tolist()

    k = 5  # liczba sąsiadów

    # 4. Nasz "czysty" Pythonowy k-NN
    abdo_knn = ABDO_KNN_Classifier(k=k)
    abdo_knn.fit(X_train_list, y_train_list)
    y_pred_abdo = abdo_knn.predict(X_test_list)
    acc_abdo = accuracy_score(y_test_list, y_pred_abdo)

    # 5. k-NN ze scikit-learn
    # skl_knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    # skl_knn.fit(X_train, y_train)
    # y_pred_skl = skl_knn.predict(X_test)
    # acc_skl = accuracy_score(y_test, y_pred_skl)

    # 6. Wyniki
    print(f"Dokładność naszego prostego k-NN (pure Python): {acc_abdo:.3f}")
    print(f"Dokładność k-NN ze scikit-learn:               {acc_skl:.3f}")

    # 7. Dla ciekawości: pierwsze kilka predykcji
    print("\nPierwsze 10 etykiet (y_test):      ", y_test_list[:10])
    print("Nasz k-NN (abdo):                ", y_pred_abdo[:10])
    print("Scikit-learn KNeighborsClassifier: ", y_pred_skl[:10])


if __name__ == "__main__":
    main()
