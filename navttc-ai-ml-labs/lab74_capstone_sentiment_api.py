# Lab 74: Capstone — End-to-end Product Review Sentiment Analyzer
# NAVTTC Week 10-12 Employable Project

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib

print("=" * 50)
print("LAB 74 CAPSTONE: Product Review Sentiment Analyzer")
print("=" * 50)

# Step 1: Business problem
print("""
Business Problem:
E-commerce platforms receive thousands of product reviews daily.
Automated sentiment analysis helps product teams prioritize fixes
and marketing teams highlight positive feedback.

Stakeholders: Product Manager, Customer Support, Data Team
Success Metric: >85% classification accuracy on held-out reviews
""")

# Step 2: Dataset
reviews = pd.DataFrame({
    "review": [
        "Battery lasts all day, camera is sharp, very satisfied",
        "Screen cracked after one week, poor build quality",
        "Fast delivery and excellent packaging",
        "Overheats during gaming, disappointed",
        "Best phone in this price range",
        "Laggy UI and slow updates",
        "Great value, smooth performance",
        "Customer support was unhelpful",
        "Love the display and sound quality",
        "Stopped charging after a month",
        "Outstanding camera for night shots",
        "Too expensive for what you get",
        "Reliable phone for daily use",
        "Apps crash frequently, frustrating",
        "Premium feel and solid design",
        "Returned it the same day",
    ],
    "sentiment": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
})
reviews["label"] = reviews["sentiment"].map({1: "positive", 0: "negative"})

print(f"Dataset size: {len(reviews)} reviews")
print(reviews["label"].value_counts())

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    reviews["review"], reviews["sentiment"],
    test_size=0.25, random_state=17, stratify=reviews["sentiment"]
)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=300)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

# Step 4: Evaluation visualization
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["neg", "pos"], yticklabels=["neg", "pos"], ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Top features
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
top_pos = np.argsort(coefs)[-5:][::-1]
top_neg = np.argsort(coefs)[:5]
top_words = list(feature_names[top_pos]) + list(feature_names[top_neg])
top_vals = list(coefs[top_pos]) + list(coefs[top_neg])
colors = ["green"] * 5 + ["red"] * 5
axes[1].barh(top_words, top_vals, color=colors)
axes[1].set_title("Top Sentiment Features")
axes[1].invert_yaxis()

plt.suptitle("Lab 74 Capstone: Sentiment Analyzer Results", fontsize=13)
plt.tight_layout()
plt.savefig("lab74_capstone_results.png", dpi=150)
print("Saved plot: lab74_capstone_results.png")

# Step 5: Save deliverables
out_dir = Path("saved_models")
out_dir.mkdir(exist_ok=True)
joblib.dump({"vectorizer": vectorizer, "model": model}, out_dir / "capstone_sentiment.pkl")

report = {
    "project": "Product Review Sentiment Analyzer",
    "accuracy": round(acc, 4),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "algorithm": "TF-IDF + Logistic Regression",
    "deployment": "saved_models/capstone_sentiment.pkl",
    "business_use": "Prioritize negative reviews for support escalation",
}
(out_dir / "capstone_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# Step 6: Live inference demo
bundle = joblib.load(out_dir / "capstone_sentiment.pkl")
demo_reviews = [
    "Amazing camera and battery life",
    "Worst purchase, broke immediately",
]

print("\nLive inference:")
for r in demo_reviews:
    vec = bundle["vectorizer"].transform([r])
    p = bundle["model"].predict(vec)[0]
    label = "positive" if p == 1 else "negative"
    print(f"  [{label.upper()}] {r}")

print(f"\nCapstone report: {out_dir / 'capstone_report.json'}")
print("Project ready for NAVTTC viva and skills competition presentation.")
