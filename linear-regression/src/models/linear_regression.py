"""
Task 1. Linear Regression from Scratch
"""
import numpy as np
from src.metrics import mse, rmse, mae, r2_score

class LinearRegression:
    def __init__(self, lr=0.01, iterations=200):
        self.lr = lr
        self.iterations = iterations
        self.w = None
        self.b = None
        self.cost_history = []


    def compute_cost(self, X: np.ndarray, y: np.ndarray, cost_func: str) -> np.float32:
        """Computing cost with the selected cost function"""
        y_pred = self.predict(X)
        if callable(cost_func):
            return cost_func(y, y_pred)
        elif cost_func == 'rmse':
            return rmse(y, y_pred)
        elif cost_func == 'mae':
            return mae(y, y_pred)
        elif cost_func == 'r2':
            return r2_score(y, y_pred)
        else:
            return mse(y, y_pred)


    def fit_gd(self, X: np.ndarray, y: np.ndarray, cost_func: str = "mse") -> None:
        """Gradient Descent Optimization"""
        n_samples, n_features = X.shape

        # Init w and b
        self.w = np.zeros((n_features, 1))
        self.b = 0
        self.cost_history = []

        # Iterate n times
        for i in range(self.iterations):
            y_pred = self.predict(X)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if i % 100 == 0 or i == self.iterations - 1:
                cost = self.compute_cost(X, y, cost_func)
                self.cost_history.append(cost)
            

    def fit_normEqn(self, X: np.ndarray, y: np.ndarray, cost_func: str = "mse") -> None:
        """Normal Equation Optimization"""
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Apply Normal Equation: theta = (X^T * X)^-1 * X^T * y
        theta_best = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.b = theta_best[0][0]
        self.w = theta_best[1:]
        cost = self.compute_cost(X, y, cost_func)
        self.cost_history.append(cost)


    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b