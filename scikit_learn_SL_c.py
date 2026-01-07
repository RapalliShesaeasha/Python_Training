from sklearn.linear_model import LogisticRegression
import numpy as np


def classification():
    
# Features: [orders_count, total_spend]
    X = np.array([
        [5, 500],
        [10, 2000],
        [2, 150],
        [8, 1200]
    ])

    # Labels: 1 = Premium, 0 = Regular
    y = np.array([0, 1, 0, 1])

    # Create model
    model = LogisticRegression()

    # Train model
    model.fit(X, y)

    # Predict new customer
    prediction = model.predict([[6, 1500]])

    print("Customer Type:", "Premium" if prediction[0] == 1 else "Regular")


if __name__ == "__main__":
    classification()
