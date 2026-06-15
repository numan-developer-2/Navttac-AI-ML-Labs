# Lab 68: Gradient descent and polynomial regression (NAVTTC Week 4-5)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(17)

print("-" * 42)
print("Lab 68: Gradient Descent and Polynomial Regression")
print("-" * 42)


def gradient_descent_univariate(x, y, lr=0.01, epochs=1000):
    """Manual gradient descent for y = w*x + b."""
    w, b = 0.0, 0.0
    n = len(x)
    history = []
    for _ in range(epochs):
        y_pred = w * x + b
        dw = (-2 / n) * np.sum(x * (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)
        w -= lr * dw
        b -= lr * db
        mse = np.mean((y - y_pred) ** 2)
        history.append(mse)
    return w, b, history


# Part 1: Univariate linear regression with gradient descent
x = np.linspace(0, 10, 50)
y = 2.5 * x + 7 + np.random.normal(0, 2, 50)

w, b, loss_history = gradient_descent_univariate(x, y, lr=0.01, epochs=800)
print(f"Learned parameters: weight={w:.3f}, bias={b:.3f}")
print(f"Final MSE: {loss_history[-1]:.4f}")

# Part 2: Polynomial regression with scikit-learn
x_poly = x.reshape(-1, 1)
y_nonlinear = 0.5 * x ** 2 - 2 * x + 10 + np.random.normal(0, 3, 50)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(x_poly.reshape(-1, 1))

model = LinearRegression()
model.fit(X_poly, y_nonlinear)
y_hat = model.predict(X_poly)

print(f"\nPolynomial R2: {r2_score(y_nonlinear, y_hat):.4f}")
print(f"Polynomial MSE: {mean_squared_error(y_nonlinear, y_hat):.4f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(x, y, alpha=0.6, label="data")
axes[0].plot(x, w * x + b, color="red", linewidth=2, label="GD fit")
axes[0].set_title("Univariate Linear (Gradient Descent)")
axes[0].legend()

axes[1].plot(loss_history, color="green")
axes[1].set_title("Loss Curve (MSE per epoch)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE")

axes[2].scatter(x, y_nonlinear, alpha=0.6, label="data")
axes[2].plot(x, y_hat, color="purple", linewidth=2, label="poly fit")
axes[2].set_title("Polynomial Regression (degree=2)")
axes[2].legend()

plt.suptitle("Lab 68: Gradient Descent and Polynomial Regression", fontsize=13)
plt.tight_layout()
plt.savefig("lab68_gradient_descent_polynomial.png", dpi=150)
print("Saved plot: lab68_gradient_descent_polynomial.png")
