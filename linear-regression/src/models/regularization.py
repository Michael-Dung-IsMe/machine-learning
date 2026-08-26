"""
Regularized Linear Regression Models implemented in pure NumPy.

Includes:
- RidgeRegression: L2 Regularization (Closed-form Analytical & Gradient Descent)
- LassoRegression: L1 Regularization (Coordinate Descent with Soft-Thresholding)
- Helper functions for coefficient paths across varying regularizing parameters (alphas)
"""

import numpy as np
from typing import Tuple, List, Optional
from src.metrics import mse, r2_score


def soft_threshold(rho: float, lambda_val: float) -> float:
    """
    Soft-thresholding operator S(rho, lambda_val) used in Lasso Coordinate Descent.
    S(rho, lambda) = sign(rho) * max(0, |rho| - lambda)
    """
    if rho > lambda_val:
        return rho - lambda_val
    elif rho < -lambda_val:
        return rho + lambda_val
    else:
        return 0.0


class RidgeRegression:
    """
    Ridge Regression (L2 Regularized Linear Regression).
    Objective: Min 1/(2N) * ||Xw + b - y||^2_2 + (alpha / 2) * ||w||^2_2
    """
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, max_iter: int = 1000, lr: float = 0.01):
        """
        Args:
            alpha: Regularization strength (must be non-negative)
            fit_intercept: Whether to calculate intercept b
            max_iter: Maximum iterations for gradient descent
            lr: Learning rate for gradient descent
        """
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.lr = lr
        self.w = None
        self.b = 0.0
        self.cost_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = "closed_form") -> "RidgeRegression":
        """
        Fits Ridge model using either closed-form analytical solution or gradient descent.
        
        Args:
            X: Feature matrix of shape (N, d)
            y: Target array of shape (N, 1) or (N,)
            method: 'closed_form' or 'gd'
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n_samples, n_features = X.shape

        if method == "closed_form":
            if self.fit_intercept:
                X_mean = np.mean(X, axis=0, keepdims=True)
                y_mean = np.mean(y, axis=0, keepdims=True)
                X_centered = X - X_mean
                y_centered = y - y_mean
            else:
                X_mean = np.zeros((1, n_features))
                y_mean = np.zeros((1, 1))
                X_centered = X
                y_centered = y

            # Analytical Closed-form: (X_c^T * X_c + alpha * N * I)^-1 * X_c^T * y_c
            # (Note: alpha * N scales regularization relative to MSE loss formulation 1/(2N))
            A = np.dot(X_centered.T, X_centered) + (self.alpha * n_samples) * np.eye(n_features)
            self.w = np.linalg.solve(A, np.dot(X_centered.T, y_centered))

            if self.fit_intercept:
                self.b = float((y_mean - np.dot(X_mean, self.w)).item())
            else:
                self.b = 0.0

            # Record final cost
            y_pred = self.predict(X)
            cost = 0.5 * mse(y, y_pred) + 0.5 * self.alpha * np.sum(self.w ** 2)
            self.cost_history = [cost]

        elif method == "gd":
            self.w = np.zeros((n_features, 1))
            self.b = 0.0
            self.cost_history = []

            for i in range(self.max_iter):
                y_pred = self.predict(X)
                error = y_pred - y

                dw = (1.0 / n_samples) * np.dot(X.T, error) + self.alpha * self.w
                db = (1.0 / n_samples) * np.sum(error) if self.fit_intercept else 0.0

                self.w -= self.lr * dw
                if self.fit_intercept:
                    self.b -= self.lr * db

                cost = 0.5 * mse(y, y_pred) + 0.5 * self.alpha * np.sum(self.w ** 2)
                self.cost_history.append(cost)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'closed_form' or 'gd'.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts output using linear model: y_pred = Xw + b"""
        X = np.asarray(X, dtype=np.float64)
        if self.w is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        return np.dot(X, self.w) + self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns R2 score on test/validation set."""
        y_pred = self.predict(X)
        return float(r2_score(y, y_pred))


class LassoRegression:
    """
    LASSO Regression (L1 Regularized Linear Regression via Coordinate Descent).
    Objective: Min 1/(2N) * ||Xw + b - y||^2_2 + alpha * ||w||_1
    """
    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 300,
        tol: float = 1e-3
    ):
        """
        Args:
            alpha: L1 regularization strength
            fit_intercept: Whether to calculate intercept b
            max_iter: Maximum coordinate descent iterations
            tol: Convergence tolerance for coefficient updates
        """
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = 0.0
        self.cost_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        """
        Fits LASSO model using Coordinate Descent algorithm.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros((n_features, 1))
        self.b = float(np.mean(y)) if self.fit_intercept else 0.0

        # Precompute column squared norms z_j = sum(X_ij^2) / N
        z = np.sum(X ** 2, axis=0, keepdims=True).T / n_samples
        # Avoid division by zero for constant features
        z[z == 0] = 1.0

        self.cost_history = []

        # Residual vector r = y - (Xw + b)
        r = (y - (np.dot(X, self.w) + self.b)).copy()

        for it in range(self.max_iter):
            w_old = self.w.copy()

            for j in range(n_features):
                # rho_j = (1/N) * X_j^T * r + w_j * z_j
                rho_j = float(np.dot(X[:, j], r[:, 0])) / n_samples + self.w[j, 0] * z[j, 0]

                # Soft thresholding update
                new_w_j = soft_threshold(rho_j, self.alpha) / z[j, 0]
                dw = new_w_j - self.w[j, 0]

                if dw != 0.0:
                    r[:, 0] -= dw * X[:, j]
                    self.w[j, 0] = new_w_j

            # Update intercept b
            if self.fit_intercept:
                b_new = float(np.mean(r[:, 0] + self.b))
                r[:, 0] -= (b_new - self.b)
                self.b = b_new

            # Calculate cost
            l1_penalty = self.alpha * np.sum(np.abs(self.w))
            cost = 0.5 * float(np.mean(r[:, 0] ** 2)) + l1_penalty
            self.cost_history.append(cost)

            # Check convergence
            max_change = np.max(np.abs(self.w - w_old))
            if max_change < self.tol:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts output using linear model: y_pred = Xw + b"""
        X = np.asarray(X, dtype=np.float64)
        if self.w is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        return np.dot(X, self.w) + self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns R2 score on test/validation set."""
        y_pred = self.predict(X)
        return float(r2_score(y, y_pred))


def compute_coefficient_paths(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes coefficient paths across a spectrum of alpha values for regularization diagnostic plots.

    Args:
        model_class: RidgeRegression or LassoRegression class
        X: Feature matrix (N, d)
        y: Target array
        alphas: Array of alpha regularizing parameters
        **kwargs: Additional parameters passed to model constructor

    Returns:
        coefs: Matrix of shape (len(alphas), d) containing weights for each alpha
        intercepts: Array of shape (len(alphas),) containing intercepts
    """
    coefs = []
    intercepts = []

    for alpha in alphas:
        model = model_class(alpha=alpha, **kwargs)
        model.fit(X, y)
        coefs.append(model.w.ravel())
        intercepts.append(model.b)

    return np.array(coefs), np.array(intercepts)