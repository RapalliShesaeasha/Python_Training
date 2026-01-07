from sklearn.linear_model import LinearRegression
import numpy as np


def linear_regression():
    # Feature (quantity sold)
    X = np.array([[1], [2], [3]])

    # Target (sales amount)
    y = np.array([100, 200, 300])

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(X, y)

    # Predict
    predicted = model.predict([[5]])

    print("Predicted sales for quantity 5:", predicted[0])


if __name__ == "__main__":
    linear_regression()
