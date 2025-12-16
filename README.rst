.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/kNN.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/kNN
    .. image:: https://readthedocs.org/projects/kNN/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://kNN.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/kNN/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/kNN
    .. image:: https://img.shields.io/pypi/v/kNN.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/kNN/
    .. image:: https://img.shields.io/conda/vn/conda-forge/kNN.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/kNN
    .. image:: https://pepy.tech/badge/kNN/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/kNN
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/kNN

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===
kNN
===

This project presents a Python implementation of the k-Nearest Neighbors (k-NN) classifier and a comparison with the "KNeighborsClassifier" from scikit-learn.

The main goals of the project are:

1. understanding the k-NN algorithm,

2. comparing classification quality (accuracy),

3. comparing execution time,

4. visualizing results (plots and confusion matrices).


k-NN Classifier - Custom implementation vs scikit-learn
=============================================

This project presents a **pure Python implementation of the k-Nearest Neighbors (k-NN) classifier**
and compares it with the reference implementation from **scikit-learn**.

The main goal is to understand how k-NN works internally and to compare:

1. classification accuracy,
2. behavior for different values of *k*,
3. execution time

The project follows a **PyScaffold-style structure**.

--------------------------------------------------

Project Structure
-----------------

::

    src/knn/
        abdo_knn.py        Custom Python implementation of k-NN
    tests/
        compare_knn.py     Accuracy and execution time comparison
        plot_knn_results.py  Accuracy plots and confusion matrices

--------------------------------------------------

Implemented Classifier
----------------------

**ABDO KNN Classifier** (`ABDOKNNClassifier`) is implemented in pure Python:

- Euclidean distance computation,
- manual distance sorting,
- majority voting using ``Counter``,
- support for multi-class classification,
- no NumPy used inside the algorithm logic.

--------------------------------------------------

Reference Implementation
------------------------

For comparison, the project uses:

- ``KNeighborsClassifier`` from **scikit-learn**

This implementation is highly optimized and serves as a performance and accuracy reference.

--------------------------------------------------

Datasets
--------

Experiments are performed using datasets from ``sklearn.datasets``:

- Iris
- Digits

--------------------------------------------------

Experiments and Visualization
-----------------------------

The project includes:

- accuracy comparison for *k = 1 … 16*,
- accuracy vs *k* plots,
- confusion matrices shown side by side,
- execution time comparison for both classifiers.

Visualization is done using **matplotlib** and **scikit-learn metrics**.

--------------------------------------------------

How to Run
----------

Run accuracy and timing comparison:

::

    python tests/compare_knn.py

Generate plots and confusion matrices:

::

    python tests/plot_knn_results.py

--------------------------------------------------

Typical Results
---------------

- Both classifiers achieve similar accuracy.
- The scikit-learn implementation is **significantly faster**.
- Larger values of *k* improve stability but may reduce sensitivity.

--------------------------------------------------

Tools used
-----------------

- Python 3
- PyScaffold
- VirtualEnv
- scikit-learn
- matplotlib

--------------------------------------------------

Documentation can be found in docs folder, named **Antoni_Bajor_kNN_documentation.pdf**

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
