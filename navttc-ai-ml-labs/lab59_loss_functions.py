# Lab 59: Common loss functions used in neural networks

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt

print("-" * 42)
print("Lab 59: Loss Functions")
print("-" * 42)

print("Loss functions score prediction error; optimizers minimize them during training.\n")

# Regression losses
actual_reg = np.array([3.0, 5.0, 2.5, 7.0])
predicted_reg = np.array([2.5, 5.0, 4.0, 8.0])

mse_value = np.mean((actual_reg - predicted_reg) ** 2)
mae_value = np.mean(np.abs(actual_reg - predicted_reg))

print("1) Mean Squared Error (regression)")
print(f"   actual={actual_reg} predicted={predicted_reg}")
print(f"   MSE = {mse_value:.4f}")

print("\n2) Mean Absolute Error (regression)")
print(f"   MAE = {mae_value:.4f}")

# Binary classification
y_true_bin = np.array([1, 0, 1, 1])
y_prob_bin = np.array([0.9, 0.1, 0.8, 0.4])
bce_value = -np.mean(
    y_true_bin * np.log(y_prob_bin + 1e-9)
    + (1 - y_true_bin) * np.log(1 - y_prob_bin + 1e-9)
)
print("\n3) Binary cross-entropy")
print(f"   BCE = {bce_value:.4f}")

# Multi-class
y_true_mc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_prob_mc = np.array([
    [0.9, 0.05, 0.05],
    [0.1, 0.8, 0.1],
    [0.2, 0.2, 0.6],
])
cce_value = -np.mean(np.sum(y_true_mc * np.log(y_prob_mc + 1e-9), axis=1))
print("\n4) Categorical cross-entropy")
print(f"   CCE = {cce_value:.4f}")

# Hinge loss
y_sign = np.array([1, -1, 1, -1])
y_margin = np.array([0.8, -0.6, -0.3, 0.5])
hinge_value = np.mean(np.maximum(0, 1 - y_sign * y_margin))
print("\n5) Hinge loss (SVM-style)")
print(f"   Hinge = {hinge_value:.4f}")

# Huber loss
delta = 1.0
residuals = actual_reg - predicted_reg
huber_parts = np.where(
    np.abs(residuals) <= delta,
    0.5 * residuals ** 2,
    delta * (np.abs(residuals) - 0.5 * delta),
)
huber_value = np.mean(huber_parts)
print("\n6) Huber loss (robust regression)")
print(f"   Huber = {huber_value:.4f}")

# KL divergence
p_dist = np.array([0.4, 0.3, 0.3])
q_dist = np.array([0.3, 0.4, 0.3])
kl_value = np.sum(p_dist * np.log(p_dist / q_dist))
print("\n7) KL divergence")
print(f"   KL = {kl_value:.4f}")

print("\nSummary")
print(f"{'Loss':<28} {'Value':>8}")
print("-" * 38)
for name, val in [
    ("MSE", mse_value),
    ("MAE", mae_value),
    ("Binary CE", bce_value),
    ("Categorical CE", cce_value),
    ("Hinge", hinge_value),
    ("Huber", huber_value),
    ("KL Divergence", kl_value),
]:
    print(f"{name:<28} {val:>8.4f}")

error_axis = np.linspace(-3, 3, 200)
mse_curve = error_axis ** 2
mae_curve = np.abs(error_axis)
huber_curve = np.where(
    np.abs(error_axis) <= 1,
    0.5 * error_axis ** 2,
    np.abs(error_axis) - 0.5,
)

plt.figure(figsize=(9, 5))
plt.plot(error_axis, mse_curve, label="MSE", linewidth=2)
plt.plot(error_axis, mae_curve, label="MAE", linewidth=2)
plt.plot(error_axis, huber_curve, label="Huber", linewidth=2, linestyle="--")
plt.xlabel("Prediction error")
plt.ylabel("Loss")
plt.title("Lab 59: MSE vs MAE vs Huber")
plt.legend()
plt.ylim(0, 6)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lab59_loss_comparison.png", dpi=150)
print("\nSaved plot: lab59_loss_comparison.png")
