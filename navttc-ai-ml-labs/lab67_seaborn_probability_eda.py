# Lab 67: Seaborn EDA and probability visualization (NAVTTC Week 3-4)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(17)
sns.set_theme(style="whitegrid")

print("-" * 42)
print("Lab 67: Seaborn EDA and Probability")
print("-" * 42)

# Synthetic student performance dataset
df = pd.DataFrame({
    "student": ["Hamza", "Ayesha", "Bilal", "Fatima", "Usman", "Hina", "Omar", "Sana"],
    "math": [78, 92, 71, 95, 84, 88, 76, 90],
    "ai_lab": [82, 91, 76, 94, 87, 89, 80, 93],
    "attendance": [88, 95, 72, 98, 85, 91, 78, 94],
    "city": ["Lahore", "Karachi", "Multan", "Lahore", "Islamabad", "Karachi", "Multan", "Lahore"],
})

print("Dataset preview:")
print(df.head())

print("\nDescriptive statistics:")
print(df[["math", "ai_lab", "attendance"]].describe().round(2))

# Correlation
corr = df[["math", "ai_lab", "attendance"]].corr()
print("\nCorrelation matrix:")
print(corr.round(3))

# Probability: normal distribution example
mu, sigma = df["ai_lab"].mean(), df["ai_lab"].std()
p_above_85 = 1 - stats.norm.cdf(85, mu, sigma)
print(f"\nP(ai_lab > 85) assuming normal dist: {p_above_85:.2%}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(df["ai_lab"], kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Distribution: AI Lab Scores")

sns.boxplot(data=df, x="city", y="ai_lab", ax=axes[0, 1])
axes[0, 1].set_title("AI Lab Scores by City")

sns.scatterplot(data=df, x="math", y="ai_lab", hue="city", s=100, ax=axes[1, 0])
axes[1, 0].set_title("Math vs AI Lab Score")

sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1, 1])
axes[1, 1].set_title("Feature Correlation Heatmap")

plt.suptitle("Lab 67: Seaborn Exploratory Data Analysis", fontsize=14)
plt.tight_layout()
plt.savefig("lab67_seaborn_eda.png", dpi=150)
print("\nSaved plot: lab67_seaborn_eda.png")

# Joint plot saved separately
g = sns.jointplot(data=df, x="math", y="ai_lab", kind="reg", height=5)
g.fig.suptitle("Lab 67: Joint distribution with regression", y=1.02)
g.savefig("lab67_jointplot.png", dpi=150)
print("Saved plot: lab67_jointplot.png")
