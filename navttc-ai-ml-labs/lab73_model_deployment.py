# Lab 73: Model persistence and deployment pattern (NAVTTC NLP deployment)

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("-" * 42)
print("Lab 73: Model Persistence and Deployment")
print("-" * 42)

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

# Train a lightweight sentiment model
texts = [
    "excellent product", "love it", "great quality", "amazing service",
    "terrible item", "hate it", "poor quality", "awful experience",
    "good enough", "not bad", "average product", "okay delivery",
] * 3
labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0] * 3

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=200)),
    ("clf", LogisticRegression(max_iter=500)),
])
pipeline.fit(texts, labels)

preds = pipeline.predict(texts)
print(f"Training accuracy: {accuracy_score(labels, preds) * 100:.2f}%")
print(classification_report(labels, preds, target_names=["negative", "positive"]))

# Save model artifacts
model_path = MODEL_DIR / "sentiment_pipeline.joblib"
meta_path = MODEL_DIR / "sentiment_metadata.json"

joblib.dump(pipeline, model_path)
metadata = {
    "model_type": "TF-IDF + LogisticRegression",
    "task": "binary_sentiment",
    "classes": ["negative", "positive"],
    "version": "1.0",
    "framework": "scikit-learn",
}
meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(f"\nSaved model: {model_path}")
print(f"Saved metadata: {meta_path}")

# Simulate deployment inference script
def predict_sentiment(review: str) -> dict:
    loaded = joblib.load(model_path)
    label = int(loaded.predict([review])[0])
    prob = loaded.predict_proba([review])[0]
    return {
        "review": review,
        "label": metadata["classes"][label],
        "confidence": float(max(prob)),
    }


samples = [
    "This phone is fantastic and fast",
    "Battery dies in one hour, very bad",
]
print("\nDeployment inference demo:")
for s in samples:
    result = predict_sentiment(s)
    print(f"  {result}")

# Export simple REST-style handler pattern (documentation)
handler_code = '''
# Example Flask deployment (install flask separately)
# from flask import Flask, request, jsonify
# import joblib
# app = Flask(__name__)
# model = joblib.load("saved_models/sentiment_pipeline.joblib")
#
# @app.post("/predict")
# def predict():
#     text = request.json.get("text", "")
#     pred = model.predict([text])[0]
#     return jsonify({"sentiment": "positive" if pred == 1 else "negative"})
'''
print("\nREST deployment pattern:")
print(handler_code.strip())
print("\nLab 73 complete: train -> save -> load -> infer")
