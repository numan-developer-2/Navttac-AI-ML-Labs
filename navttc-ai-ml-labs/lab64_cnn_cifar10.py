# Lab 64: Deep CNN on CIFAR-10 color images

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

print("-" * 42)
print("Lab 64: CNN on CIFAR-10")
print("-" * 42)

print(
    "CIFAR-10 has 60k RGB images (32x32) across 10 object classes.\n"
    "Color and background variation make it harder than MNIST.\n"
)

(train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
class_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

train_x = train_x.astype("float32") / 255.0
test_x = test_x.astype("float32") / 255.0
train_y_oh = keras.utils.to_categorical(train_y, 10)
test_y_oh = keras.utils.to_categorical(test_y, 10)

print(f"Train: {train_x.shape} | Test: {test_x.shape}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(train_x[i])
    ax.set_title(class_labels[train_y[i][0]], fontsize=9)
    ax.axis("off")
plt.suptitle("Lab 64: CIFAR-10 sample images", fontsize=13)
plt.tight_layout()
plt.savefig("lab64_cifar10_samples.png", dpi=150)
print("Saved plot: lab64_cifar10_samples.png")

cifar_cnn = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.3),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation="softmax"),
])
cifar_cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nModel summary:")
cifar_cnn.summary()

history = cifar_cnn.fit(
    train_x, train_y_oh,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1,
)

loss, acc = cifar_cnn.evaluate(test_x, test_y_oh, verbose=0)
print(f"\nTest accuracy: {acc * 100:.2f}%")
print(f"Test loss: {loss:.4f}")

pred_idx = np.argmax(cifar_cnn.predict(test_x, verbose=0), axis=1)
true_idx = test_y.flatten()

print("\nPer-class accuracy:")
for i, label in enumerate(class_labels):
    mask = true_idx == i
    class_acc = np.mean(pred_idx[mask] == true_idx[mask])
    bar = "#" * int(class_acc * 20)
    print(f"  {label:<12} {bar:<20} {class_acc * 100:.1f}%")

fig2, axes2 = plt.subplots(3, 5, figsize=(13, 8))
for i, ax in enumerate(axes2.flat):
    ax.imshow(test_x[i])
    p, t = pred_idx[i], true_idx[i]
    ax.set_title(f"P:{class_labels[p]}\nA:{class_labels[t]}", color="green" if p == t else "red", fontsize=7)
    ax.axis("off")
plt.suptitle("Lab 64: CIFAR-10 predictions", fontsize=12)
plt.tight_layout()
plt.savefig("lab64_cifar10_predictions.png", dpi=150)
print("Saved plot: lab64_cifar10_predictions.png")

fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4))
axes3[0].plot(history.history["accuracy"], label="train")
axes3[0].plot(history.history["val_accuracy"], label="validation")
axes3[0].set_title("Accuracy")
axes3[0].set_xlabel("Epoch")
axes3[0].legend()

axes3[1].plot(history.history["loss"], label="train")
axes3[1].plot(history.history["val_loss"], label="validation")
axes3[1].set_title("Loss")
axes3[1].set_xlabel("Epoch")
axes3[1].legend()

plt.suptitle("Lab 64: CIFAR-10 training history", fontsize=13)
plt.tight_layout()
plt.savefig("lab64_cifar10_training.png", dpi=150)
print("Saved plot: lab64_cifar10_training.png")
