# Lab 69: Ensemble learning — Random Forest and XGBoost (NAVTTC Week 6-7)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

print("-" * 42)
print("Lab 69: Ensemble ML (Bagging and Boosting)")
print("-" * 42)

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=17, stratify=data.target
)

# Bagging: Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=17)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"Random Forest accuracy: {rf_acc * 100:.2f}%")
print("\nRandom Forest report:")
print(classification_report(y_test, rf_pred, target_names=data.target_names))

# Boosting: XGBoost
if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=17,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"\nXGBoost accuracy: {xgb_acc * 100:.2f}%")
    print("\nXGBoost report:")
    print(classification_report(y_test, xgb_pred, target_names=data.target_names))
else:
    print("\nInstall xgboost for boosting demo: pip install xgboost")

# Feature importance
importances = rf.feature_importances_
top_idx = np.argsort(importances)[-10:]

fig, axes = plt.subplots(1, 2 if HAS_XGB else 1, figsize=(12, 5))
if not isinstance(axes, np.ndarray):
    axes = [axes]

sns.heatmap(
    confusion_matrix(y_test, rf_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names,
    ax=axes[0],
)
axes[0].set_title("Random Forest Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

if HAS_XGB:
    sns.heatmap(
        confusion_matrix(y_test, xgb_pred),
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=data.target_names,
        yticklabels=data.target_names,
        ax=axes[1],
    )
    axes[1].set_title("XGBoost Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

plt.suptitle("Lab 69: Ensemble Methods", fontsize=13)
plt.tight_layout()
plt.savefig("lab69_ensemble_confusion.png", dpi=150)
print("Saved plot: lab69_ensemble_confusion.png")

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(
    [data.feature_names[i] for i in top_idx],
    importances[top_idx],
    color="teal",
)
ax2.set_title("Lab 69: Top 10 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("lab69_feature_importance.png", dpi=150)
print("Saved plot: lab69_feature_importance.png")
