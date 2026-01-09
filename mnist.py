import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist


def mnist_model():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values (0–255 → 0–1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Build model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    model.fit(x_train, y_train, epochs=5)

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    mnist_model()
