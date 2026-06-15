# Lab 58: Feedforward network for binary diabetes-style classification

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow import keras

print("-" * 42)
print("Lab 58: Feedforward Neural Network")
print("-" * 42)

print(
    "Feedforward nets pass data forward only: input -> hidden layers -> output.\n"
    "No recurrent connections; each layer transforms the previous activation.\n"
)

np.random.seed(17)
sample_count = 768

glucose = np.random.normal(120, 30, sample_count)
bmi = np.random.normal(32, 7, sample_count)
age = np.random.normal(33, 11, sample_count)
blood_pressure = np.random.normal(70, 18, sample_count)
skin = np.random.normal(20, 15, sample_count)
insulin = np.random.normal(80, 100, sample_count)
pregnancies = np.random.randint(0, 10, sample_count)
pedigree = np.random.uniform(0.1, 2.4, sample_count)

feature_matrix = np.column_stack([
    glucose, bmi, age, blood_pressure, skin, insulin, pregnancies, pedigree
])

risk = 1 / (1 + np.exp(-(-5 + 0.03 * glucose + 0.05 * bmi)))
diabetes_label = (np.random.uniform(0, 1, sample_count) < risk).astype(int)

print(f"Samples: {sample_count}")
print(f"Positive class: {diabetes_label.sum()} | Negative class: {(diabetes_label == 0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, diabetes_label, test_size=0.2, random_state=17
)

feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)

ff_network = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
ff_network.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\nNetwork summary:")
ff_network.summary()

history = ff_network.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.15,
    verbose=0,
)

loss, accuracy = ff_network.evaluate(X_test, y_test, verbose=0)
y_hat = (ff_network.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

print(f"\nTest accuracy: {accuracy * 100:.2f}%")
print(f"Test loss: {loss:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_hat, target_names=["No Diabetes", "Diabetes"]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
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

plt.suptitle("Lab 58: Feedforward network training", fontsize=13)
plt.tight_layout()
plt.savefig("lab58_feedforward_training.png", dpi=150)
print("Saved plot: lab58_feedforward_training.png")
