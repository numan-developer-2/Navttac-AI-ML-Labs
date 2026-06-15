# Lab 63: Convolutional neural network on MNIST

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

print("-" * 42)
print("Lab 63: Convolutional Neural Network")
print("-" * 42)

print(
    "CNNs use convolution + pooling to learn spatial patterns in images.\n"
    "Typical stack: Conv2D -> MaxPool -> Conv2D -> Flatten -> Dense.\n"
)

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

train_x = train_x.reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_x = test_x.reshape(-1, 28, 28, 1).astype("float32") / 255.0
train_y_oh = keras.utils.to_categorical(train_y, 10)
test_y_oh = keras.utils.to_categorical(test_y, 10)

print(f"Train tensor shape: {train_x.shape}")
print(f"Test tensor shape : {test_x.shape}")

cnn = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax"),
])
cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("\nModel summary:")
cnn.summary()

history = cnn.fit(
    train_x, train_y_oh,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
)

loss, acc = cnn.evaluate(test_x, test_y_oh, verbose=0)
print(f"\nTest accuracy: {acc * 100:.2f}%")
print(f"Test loss: {loss:.4f}")

# Visualize first conv layer filters on one image
sample = test_x[0:1]
input_layer = keras.Input(shape=(28, 28, 1))
feature_extractor = keras.Model(inputs=input_layer, outputs=cnn.layers[0](input_layer))
feature_maps = feature_extractor.predict(sample, verbose=0)

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    if i < feature_maps.shape[-1]:
        ax.imshow(feature_maps[0, :, :, i], cmap="viridis")
    ax.axis("off")
plt.suptitle("Lab 63: First conv layer feature maps", fontsize=12)
plt.tight_layout()
plt.savefig("lab63_cnn_feature_maps.png", dpi=150)
print("Saved plot: lab63_cnn_feature_maps.png")

pred_labels = np.argmax(cnn.predict(test_x[:10], verbose=0), axis=1)
fig2, axes2 = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes2.flat):
    ax.imshow(test_x[i].reshape(28, 28), cmap="gray")
    pred, truth = pred_labels[i], test_y[i]
    ax.set_title(f"P:{pred} A:{truth}", color="green" if pred == truth else "red", fontsize=9)
    ax.axis("off")
plt.suptitle("Lab 63: CNN predictions on MNIST", fontsize=12)
plt.tight_layout()
plt.savefig("lab63_cnn_predictions.png", dpi=150)
print("Saved plot: lab63_cnn_predictions.png")
