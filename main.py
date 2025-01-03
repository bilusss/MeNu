import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

# Loading MNIST data (database of handwritten numbers)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Data normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

# Creating the ANN model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Model compilation
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
# Training the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=32,
    verbose=2
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}\nTest loss: {test_loss:.4f}")