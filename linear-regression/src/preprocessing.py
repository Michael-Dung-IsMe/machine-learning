"""
Dataset Preprocessing & Scaling Module for Bike Sharing Dataset and General ML Tasks.

Includes:
- Pure NumPy Scalers: StandardScaler, MinMaxScaler
- Data Splitting: train_test_split, train_val_test_split (supports both random and chronological/temporal splits)
- Bike Sharing Data Loading & Preprocessing: load_and_preprocess_bike_sharing
"""

import os
import zipfile
import urllib.request
from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd


class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.std_ = np.std(X, axis=0, keepdims=True)
        # Prevent division by zero for constant features
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler instance is not fitted yet. Call 'fit' before 'transform'.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler instance is not fitted yet.")
        return X_scaled * self.std_ + self.mean_


class MinMaxScaler:
    """Transforms features by scaling each feature to a given range (default 0 to 1)."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.asarray(X, dtype=np.float64)
        self.data_min_ = np.min(X, axis=0, keepdims=True)
        self.data_max_ = np.max(X, axis=0, keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("MinMaxScaler instance is not fitted yet. Call 'fit' before 'transform'.")
        
        range_diff = self.data_max_ - self.data_min_
        range_diff[range_diff == 0] = 1.0  # Prevent division by zero
        
        scale = (self.feature_range[1] - self.feature_range[0]) / range_diff
        return (X - self.data_min_) * scale + self.feature_range[0]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("MinMaxScaler instance is not fitted yet.")
        range_diff = self.data_max_ - self.data_min_
        range_diff[range_diff == 0] = 1.0
        scale = (self.feature_range[1] - self.feature_range[0]) / range_diff
        return (X_scaled - self.feature_range[0]) / scale + self.data_min_


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits arrays into train and test subsets.
    
    Args:
        X: Feature matrix
        y: Target array
        test_size: Ratio of test set (0.0 to 1.0)
        shuffle: If True, randomly shuffle before splitting. If False, split chronologically.
        random_state: Random seed for reproducibility
    """
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits arrays into train, validation, and test subsets.
    """
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_val - n_test

    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx]
    )


def load_and_preprocess_bike_sharing(
    file_path: str,
    target_col: str = "cnt",
    drop_leakage_cols: bool = True,
    as_numpy: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray, pd.DataFrame], Tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    """
    Loads and preprocesses the Bike Sharing dataset from a CSV path, ZIP file, or folder.
    
    Prevents data leakage by removing 'casual' and 'registered' columns when predicting total count 'cnt'.
    Removes non-predictive metadata ('instant', 'dteday').

    Args:
        file_path: Path to dataset (.csv, .zip, or directory containing hour.csv/day.csv)
        target_col: Target variable name (default: 'cnt')
        drop_leakage_cols: If True, drops 'casual' and 'registered' to avoid target leakage.
        as_numpy: If True, returns X and y as NumPy arrays.

    Returns:
        X: Feature matrix
        y: Target column (reshaped as 2D array if as_numpy=True)
        df_cleaned: Cleaned pandas DataFrame before matrix conversion
    """
    UCI_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

    # Download dataset if file does not exist
    if not os.path.exists(file_path):
        print(f"❗File path '{file_path}' not found. Downloading Bike Sharing dataset from UCI...")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        try:
            urllib.request.urlretrieve(UCI_URL, file_path)
            print("✅ Download completed.")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download dataset from {UCI_URL}: {e}")

    # Handle zip files
    if file_path.endswith(".zip"):
        extract_dir = os.path.join(os.path.dirname(file_path), "unzipped_bike_sharing")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Prefer hour.csv if present, else day.csv
        if os.path.exists(os.path.join(extract_dir, "hour.csv")):
            target_file = os.path.join(extract_dir, "hour.csv")
        elif os.path.exists(os.path.join(extract_dir, "day.csv")):
            target_file = os.path.join(extract_dir, "day.csv")
        else:
            raise FileNotFoundError(f"Neither 'hour.csv' nor 'day.csv' found inside {file_path}")
    elif os.path.isdir(file_path):
        if os.path.exists(os.path.join(file_path, "hour.csv")):
            target_file = os.path.join(file_path, "hour.csv")
        elif os.path.exists(os.path.join(file_path, "day.csv")):
            target_file = os.path.join(file_path, "day.csv")
        else:
            raise FileNotFoundError(f"Neither 'hour.csv' nor 'day.csv' found in folder {file_path}")
    else:
        target_file = file_path

    # Read CSV
    df = pd.read_csv(target_file)

    # Clean data
    df_cleaned = df.copy()

    # Drop index and date string columns
    cols_to_drop = ["instant", "dteday"]
    
    # Prevent Target Leakage: casual + registered == cnt
    if drop_leakage_cols and target_col == "cnt":
        cols_to_drop.extend(["casual", "registered"])
    elif drop_leakage_cols and target_col in ["casual", "registered"]:
        cols_to_drop.extend([c for c in ["casual", "registered", "cnt"] if c != target_col])

    # Drop existing columns from the list
    cols_to_drop = [c for c in cols_to_drop if c in df_cleaned.columns]
    df_cleaned = df_cleaned.drop(columns=cols_to_drop)

    # Separate X and y
    if target_col not in df_cleaned.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset columns: {list(df_cleaned.columns)}")

    X_df = df_cleaned.drop(columns=[target_col])
    y_df = df_cleaned[target_col]

    if as_numpy:
        X = X_df.values.astype(np.float64)
        y = y_df.values.astype(np.float64).reshape(-1, 1)
        return X, y, df_cleaned
    
    return X_df, y_df, df_cleaned