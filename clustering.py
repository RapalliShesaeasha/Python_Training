from sklearn.cluster import KMeans
import numpy as np


def clustering():
    # Customer data: [orders_count, total_spend]
    X = np.array([
        [2, 200],
        [3, 250],
        [10, 2000],
        [12, 2300],
        [5, 500],
        [8, 1200]
    ])

    # Create KMeans model
    model = KMeans(n_clusters=2, random_state=42)

    # Fit model
    model.fit(X)

    # Predict clusters
    labels = model.predict(X)

    print("Cluster labels:", labels)


if __name__ == "__main__":
    clustering()
