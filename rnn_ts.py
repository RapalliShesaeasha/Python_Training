import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense


def rnn_time_series():
    # Time series data
    X = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ])

    y = np.array([4, 5, 6, 7])

    # Reshape for RNN: (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build model (LSTM)
    model = Sequential([
        GRU(50, activation="relu", input_shape=(3, 1)),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    # Train model
    model.fit(X, y, epochs=200, verbose=0)

    # Predict next value
    prediction = model.predict(np.array([[[5], [6], [7]]]))
    print("Predicted next value:", prediction[0][0])


if __name__ == "__main__":
    rnn_time_series()
