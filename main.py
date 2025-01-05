import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Input
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Loading MNIST data (database of handwritten numbers)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(mnist.load_data())

# Data normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

# Function to plot sample images
def plot_sample_images(X, y):
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X[i], cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.show()

# Plotting sample images
plot_sample_images(X_train, y_train)

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()

def visualize_predictions(model, X, y):
    predictions = model.predict(X[:16])
    predicted_labels = tf.argmax(predictions, axis=1).numpy()

    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X[i], cmap='gray')
        plt.title(f"True: {y[i]}, Pred: {predicted_labels[i]}")
        plt.axis('off')
    plt.show()

# Creating the ANN model
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
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
    epochs=10,
    batch_size=32,
    verbose=2
)

# Plotting training history
plot_training_history(history)

# Visualizing predictions
visualize_predictions(model, X_test, y_test)

# Confusion matrix
y_pred = tf.argmax(model.predict(X_test), axis=1).numpy()
confusion_mtx = confusion_matrix(y_test, y_pred)

# Plotting confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='viridis', cbar=True)
plt.xlabel('Recognized values')
plt.ylabel('Values entered')
plt.title('Confusion matrix for ANN model (MNIST)')
plt.xticks(ticks=np.arange(10) + 0.5, labels=np.arange(10))
plt.yticks(ticks=np.arange(10) + 0.5, labels=np.arange(10), rotation=0)
plt.show()

# Evaluating the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}\nTest loss: {test_loss:.4f}")