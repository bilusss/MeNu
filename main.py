import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.datasets import mnist

# Loading MNIST data (database of handwritten numbers)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Data normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

# Creating the ANN model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Spłaszczenie obrazów do wektora
    Dense(128, activation='relu'),  # Warstwa ukryta
    Dense(10, activation='softmax')  # Warstwa wyjściowa
])

# Model compilation
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("The model was created and compiled.")
