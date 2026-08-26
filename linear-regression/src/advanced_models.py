"""
Advanced Models & Feature Engineering Module for Linear Regression Improvements.

Includes:
- One-Hot Encoding Transformer for Bike Sharing Dataset
- SegmentedLinearRegression (Fits separate Linear Regression models for workingday == 1 vs 0)
- Target Log Transformation Utilities
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from src.models.linear_regression import LinearRegression


def prepare_advanced_features(
    df_cleaned: pd.DataFrame,
    target_col: str = "cnt",
    use_one_hot: bool = True,
    categorical_cols: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepares features from cleaned DataFrame, optionally applying One-Hot Encoding to categorical variables.

    Args:
        df_cleaned: Cleaned pandas DataFrame containing features and target.
        target_col: Name of target column (default: 'cnt').
        use_one_hot: If True, applies One-Hot Encoding to categorical columns.
        categorical_cols: List of categorical column names to one-hot encode.

    Returns:
        X: Feature matrix as 2D NumPy array
        y: Target array as 2D NumPy array (N, 1)
        feature_names: List of feature names corresponding to columns of X
    """
    df = df_cleaned.copy()

    if categorical_cols is None:
        categorical_cols = ["hr", "mnth", "weekday", "weathersit", "season"]

    # Filter categorical cols present in df
    cat_cols_present = [c for c in categorical_cols if c in df.columns]

    if use_one_hot and cat_cols_present:
        # Apply pd.get_dummies with drop_first=True to avoid dummy variable trap
        df = pd.get_dummies(df, columns=cat_cols_present, drop_first=True, dtype=float)

    # Separate X and y
    y_df = df[target_col]
    X_df = df.drop(columns=[target_col])

    X = X_df.values.astype(np.float64)
    y = y_df.values.astype(np.float64).reshape(-1, 1)
    feature_names = list(X_df.columns)

    return X, y, feature_names


class SegmentedLinearRegression:
    """
    Segmented Linear Regression.
    Fits separate LinearRegression models for subgroups defined by a binary segment column (e.g. workingday).
    """
    def __init__(self, segment_col_idx: int):
        """
        Args:
            segment_col_idx: Index of the binary column used to segment data (e.g. workingday column index).
        """
        self.segment_col_idx = segment_col_idx
        self.model_seg1 = LinearRegression()  # e.g., workingday == 1
        self.model_seg0 = LinearRegression()  # e.g., workingday == 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SegmentedLinearRegression":
        mask1 = (X[:, self.segment_col_idx] >= 0.5)
        mask0 = ~mask1

        # Remove segment column from input matrices for sub-models to avoid zero variance
        X_sub = np.delete(X, self.segment_col_idx, axis=1)

        if np.any(mask1):
            self.model_seg1.fit_normEqn(X_sub[mask1], y[mask1])
        if np.any(mask0):
            self.model_seg0.fit_normEqn(X_sub[mask0], y[mask0])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mask1 = (X[:, self.segment_col_idx] >= 0.5)
        mask0 = ~mask1

        X_sub = np.delete(X, self.segment_col_idx, axis=1)
        y_pred = np.zeros((len(X), 1))

        if np.any(mask1):
            y_pred[mask1] = self.model_seg1.predict(X_sub[mask1])
        if np.any(mask0):
            y_pred[mask0] = self.model_seg0.predict(X_sub[mask0])

        return y_pred


def log1p_transform(y: np.ndarray) -> np.ndarray:
    """Applies log(1 + y) transformation to target array."""
    return np.log1p(np.maximum(0.0, y))


def expm1_inverse_transform(y_log: np.ndarray) -> np.ndarray:
    """Applies exp(y_log) - 1 inverse transformation and clips negative values to 0."""
    return np.maximum(0.0, np.expm1(y_log))
