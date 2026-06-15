# Lab 71: LSTM text sequence classifier (NAVTTC Week 8-10)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("-" * 42)
print("Lab 71: LSTM Sequence Model for Text")
print("-" * 42)

# Small sentiment dataset (1D sequence classification)
reviews = [
    "great product love it",
    "amazing quality fast delivery",
    "terrible waste of money",
    "worst experience ever",
    "good value for price",
    "broken after one day",
    "excellent service highly recommend",
    "very disappointed poor quality",
    "fantastic and reliable",
    "horrible customer support",
    "best purchase this year",
    "do not buy this item",
]
labels = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 1=positive

max_len = 8
vocab_size = 50

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(sequences, maxlen=max_len, padding="post")
y = labels

print(f"Sequences shape: {X.shape}")
print(f"Sample sequence: {X[0]} -> label {y[0]}")

model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=max_len),
    keras.layers.LSTM(32),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(X, y, epochs=80, verbose=0, validation_split=0.2)
loss, acc = model.evaluate(X, y, verbose=0)
print(f"\nTraining accuracy: {acc * 100:.2f}%")

# Predict on new samples
test_phrases = ["love this product", "bad quality hate it", "okay not great"]
test_seq = pad_sequences(tokenizer.texts_to_sequences(test_phrases), maxlen=max_len, padding="post")
probs = model.predict(test_seq, verbose=0).flatten()

print("\nInference:")
for phrase, p in zip(test_phrases, probs):
    sentiment = "positive" if p >= 0.5 else "negative"
    print(f"  '{phrase}' -> {sentiment} ({p:.2f})")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="validation")
axes[0].set_title("LSTM Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="validation")
axes[1].set_title("LSTM Loss")
axes[1].legend()

plt.suptitle("Lab 71: LSTM Text Classifier", fontsize=13)
plt.tight_layout()
plt.savefig("lab71_lstm_training.png", dpi=150)
print("Saved plot: lab71_lstm_training.png")
