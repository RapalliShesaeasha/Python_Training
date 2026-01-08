from sklearn.decomposition import PCA
import numpy as np


def pca():

    # Sample data: [orders, spend, visits]
    X = np.array([
        [2, 200, 5],
        [3, 250, 6],
        [10, 2000, 20],
        [12, 2300, 25],
        [5, 500, 10]
    ])

    # Reduce to 2 components
    pca = PCA(n_components=2)

    # Fit and transform
    X_reduced = pca.fit_transform(X)

    print("Reduced Data:\n", X_reduced)


if __name__ == "__main__":
    pca()
