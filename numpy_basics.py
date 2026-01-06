import numpy as np


def numpy_matrix_operations():
    # 1D Array
    arr = np.array([1, 2, 3, 4])
    print("Array:", arr)

    # 2D Array (Matrix)
    matrix = np.array([
        [10, 20, 30],
        [40, 50, 60]
    ])
    print("Matrix:\n", matrix)

    # Broadcasting
    vector = np.array([1, 2, 3])
    print("Broadcasting Result:\n", matrix + vector)

    # Vectorization
    prices = np.array([100, 200, 300])
    discounted_prices = prices * 0.9
    print("Discounted Prices:", discounted_prices)

    # Matrix multiplication
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    print("Matrix Multiplication:\n", a @ b)


if __name__ == "__main__":
    numpy_matrix_operations()
