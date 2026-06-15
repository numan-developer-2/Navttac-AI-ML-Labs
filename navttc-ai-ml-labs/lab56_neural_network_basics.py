# Lab 56: Neural network basics (XOR + Iris classification)
# Uses TensorFlow/Keras on XOR logic and the Iris dataset.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow import keras

BANNER = "-" * 42

print(BANNER)
print("Lab 56: Neural Networks")
print(BANNER)

print(
    """
Neural networks stack layers of weighted connections.
Training updates weights with backpropagation to reduce loss.
"""
)

# Part A: XOR (not linearly separable)
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
xor_targets = np.array([[0], [1], [1], [0]], dtype=float)

print("XOR truth table:")
for row, label in zip(xor_inputs, xor_targets):
    print(f"  {row.astype(int).tolist()} -> {int(label[0])}")

xor_network = keras.Sequential([
    keras.layers.Dense(4, activation="relu", input_shape=(2,)),
    keras.layers.Dense(1, activation="sigmoid"),
])
xor_network.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
xor_network.fit(xor_inputs, xor_targets, epochs=500, verbose=0)

xor_preds = (xor_network.predict(xor_inputs, verbose=0) > 0.5).astype(int)
print("\nXOR predictions:")
for row, pred, truth in zip(xor_inputs, xor_preds, xor_targets):
    mark = "OK" if pred[0] == truth[0] else "MISS"
    print(f"  {row.astype(int).tolist()} -> pred={pred[0]} actual={int(truth[0])} [{mark}]")

# Part B: Iris multi-class model
print("\nIris classification with a small feedforward network")

iris = load_iris()
features = iris.data
labels = iris.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
labels_oh = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels_oh, test_size=0.2, random_state=17
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

iris_model = keras.Sequential([
    keras.layers.Dense(16, activation="relu", input_shape=(4,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])
iris_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("\nModel summary:")
iris_model.summary()

history = iris_model.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=0)
test_loss, test_acc = iris_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.suptitle("Lab 56: Iris neural network training", fontsize=13)
plt.tight_layout()
plt.savefig("lab56_neural_network_training.png", dpi=150)
print("Saved plot: lab56_neural_network_training.png")
