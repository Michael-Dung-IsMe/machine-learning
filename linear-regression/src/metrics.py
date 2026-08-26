# Include all metrics (MSE, RMSE, MAE, R2 Score)
import numpy as np

def mse(y: np.ndarray, y_pred: np.ndarray) -> np.float32:
    return np.mean((y - y_pred) ** 2)

def rmse(y: np.ndarray, y_pred: np.ndarray) -> np.float32:
    return np.sqrt(np.mean((y - y_pred) ** 2))

def mae(y: np.ndarray, y_pred: np.ndarray) -> np.float32:
    return np.mean(np.abs(y - y_pred))

def r2_score(y: np.ndarray, y_pred: np.ndarray) -> np.float32:
    return 1 - (np.sum((y - y_pred) ** 2)) / (np.sum((y - np.mean(y)) ** 2))