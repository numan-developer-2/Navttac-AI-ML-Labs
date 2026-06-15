# Lab 57: Multi-layer perceptron on MNIST digits

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

print("-" * 42)
print("Lab 57: Multi-Layer Perceptron (MNIST)")
print("-" * 42)

print(
    "An MLP connects every neuron in one layer to the next (fully connected).\n"
    "Flow: input -> hidden dense layers -> softmax output.\n"
)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

print(f"Training images : {train_images.shape[0]}")
print(f"Test images     : {test_images.shape[0]}")
print(f"Pixel grid      : {train_images.shape[1]}x{train_images.shape[2]}")

# Normalize and flatten 28x28 -> 784
train_flat = train_images.reshape(-1, 784).astype("float32") / 255.0
test_flat = test_images.reshape(-1, 784).astype("float32") / 255.0

train_y = keras.utils.to_categorical(train_labels, 10)
test_y = keras.utils.to_categorical(test_labels, 10)

mlp = keras.Sequential([
    keras.layers.Dense(256, activation="relu", input_shape=(784,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
mlp.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("\nArchitecture:")
mlp.summary()

history = mlp.fit(
    train_flat, train_y,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
)

loss, acc = mlp.evaluate(test_flat, test_y, verbose=0)
print(f"\nTest accuracy: {acc * 100:.2f}%")
print(f"Test loss: {loss:.4f}")

sample_preds = np.argmax(mlp.predict(test_flat[:10], verbose=0), axis=1)
print("\nFirst 10 predictions:", sample_preds.tolist())
print("First 10 labels     :", test_labels[:10].tolist())

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx, ax in enumerate(axes.flat):
    ax.imshow(test_images[idx], cmap="gray")
    predicted = sample_preds[idx]
    actual = test_labels[idx]
    color = "green" if predicted == actual else "red"
    ax.set_title(f"P:{predicted} A:{actual}", color=color, fontsize=9)
    ax.axis("off")
plt.suptitle("Lab 57: MLP digit predictions", fontsize=12)
plt.tight_layout()
plt.savefig("lab57_mlp_digit_predictions.png", dpi=150)
print("Saved plot: lab57_mlp_digit_predictions.png")

fig2, ax_pair = plt.subplots(1, 2, figsize=(11, 4))
ax_pair[0].plot(history.history["accuracy"], label="train")
ax_pair[0].plot(history.history["val_accuracy"], label="validation")
ax_pair[0].set_title("Accuracy per epoch")
ax_pair[0].set_xlabel("Epoch")
ax_pair[0].legend()

ax_pair[1].plot(history.history["loss"], label="train")
ax_pair[1].plot(history.history["val_loss"], label="validation")
ax_pair[1].set_title("Loss per epoch")
ax_pair[1].set_xlabel("Epoch")
ax_pair[1].legend()

plt.suptitle("Lab 57: MLP training curves", fontsize=13)
plt.tight_layout()
plt.savefig("lab57_mlp_training_curves.png", dpi=150)
print("Saved plot: lab57_mlp_training_curves.png")
