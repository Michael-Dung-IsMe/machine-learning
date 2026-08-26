"""
Basis Expansion Module implemented in pure NumPy.

Transforms original feature representations into higher-dimensional non-linear feature spaces.

Includes:
- PolynomialBasis: Generates polynomial powers and interaction terms.
- PeriodicBasis: Generates sine and cosine trigonometric terms for cyclic features (e.g., hour, month, weekday).
- InteractionBasis: Generates pairwise interaction products (x_i * x_j).
- BikeSharingBasisExpansion: Specialized composite expansion combining periodic cyclical transformations and weather interaction terms.
"""

import itertools
import numpy as np
from typing import List, Tuple, Optional, Union


class PolynomialBasis:
    """
    Polynomial Basis Expansion.
    Generates polynomial features up to specified degree.
    """
    def __init__(self, degree: int = 2, include_interactions: bool = True, include_bias: bool = False):
        """
        Args:
            degree: Maximum polynomial degree
            include_interactions: If True, include cross terms (x_i * x_j). If False, only (x_i^p).
            include_bias: If True, include a constant term (column of 1s).
        """
        self.degree = degree
        self.include_interactions = include_interactions
        self.include_bias = include_bias
        self.combinations_ = []

    def fit(self, X: np.ndarray) -> "PolynomialBasis":
        n_samples, n_features = X.shape
        combo_list = []

        if self.include_bias:
            combo_list.append(())  # Bias term

        # Degree 1 terms (original features)
        for i in range(n_features):
            combo_list.append((i,))

        # Degree 2 to degree
        for d in range(2, self.degree + 1):
            if self.include_interactions:
                combos = itertools.combinations_with_replacement(range(n_features), d)
            else:
                combos = [(i,) * d for i in range(n_features)]
            combo_list.extend(combos)

        self.combinations_ = combo_list
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if not self.combinations_:
            self.fit(X)

        output_cols = []
        for combo in self.combinations_:
            if len(combo) == 0:
                output_cols.append(np.ones((n_samples, 1)))
            else:
                col = np.prod(X[:, combo], axis=1, keepdims=True)
                output_cols.append(col)

        return np.hstack(output_cols)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PeriodicBasis:
    """
    Periodic (Trigonometric / Fourier) Basis Expansion.
    Transforms cyclic features into sin and cos harmonics:
        sin(2 * pi * k * x / T) and cos(2 * pi * k * x / T)
    """
    def __init__(self, feature_indices: List[int], periods: List[float], n_harmonics: int = 1):
        """
        Args:
            feature_indices: Indices of columns to apply periodic transformation
            periods: Corresponding fundamental period T for each cyclic feature (e.g. 24 for hour, 7 for weekday)
            n_harmonics: Number of harmonic frequencies k = 1, 2, ..., n_harmonics
        """
        if len(feature_indices) != len(periods):
            raise ValueError("feature_indices and periods must have the same length.")
        self.feature_indices = feature_indices
        self.periods = periods
        self.n_harmonics = n_harmonics

    def fit(self, X: np.ndarray) -> "PeriodicBasis":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        expanded_features = [X]  # Keep original features

        for idx, period in zip(self.feature_indices, self.periods):
            x_col = X[:, idx:idx+1]
            for k in range(1, self.n_harmonics + 1):
                angle = 2.0 * np.pi * k * x_col / float(period)
                sin_col = np.sin(angle)
                cos_col = np.cos(angle)
                expanded_features.extend([sin_col, cos_col])

        return np.hstack(expanded_features)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class InteractionBasis:
    """
    Interaction Basis Expansion.
    Generates pairwise products between specified feature columns or all feature pairs.
    """
    def __init__(self, feature_pairs: Optional[List[Tuple[int, int]]] = None):
        """
        Args:
            feature_pairs: Specific list of tuple pairs (i, j). If None, generates all unique pairs.
        """
        self.feature_pairs = feature_pairs

    def fit(self, X: np.ndarray) -> "InteractionBasis":
        if self.feature_pairs is None:
            n_features = X.shape[1]
            self.feature_pairs = list(itertools.combinations(range(n_features), 2))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.feature_pairs is None:
            self.fit(X)

        interaction_cols = [X]  # Keep original features
        for i, j in self.feature_pairs:
            prod_col = (X[:, i:i+1] * X[:, j:j+1])
            interaction_cols.append(prod_col)

        return np.hstack(interaction_cols)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class BikeSharingBasisExpansion:
    """
    Specialized Domain Basis Expansion for Bike Sharing Dataset.
    Columns:
    0: season (period 4)
    1: yr
    2: mnth (period 12)
    3: hr (period 24)
    4: holiday
    5: weekday (period 7)
    6: workingday
    7: weathersit
    8: temp
    9: atemp
    10: hum
    11: windspeed
    """
    def __init__(self, n_harmonics: int = 2, poly_degree: int = 2):
        self.n_harmonics = n_harmonics
        self.poly_degree = poly_degree

    def fit(self, X: np.ndarray) -> "BikeSharingBasisExpansion":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        features = [X]

        # 1. Periodic Transformations for cyclic columns
        # Column 3 (hr): period 24
        # Column 2 (mnth): period 12
        # Column 5 (weekday): period 7
        # Column 0 (season): period 4
        cyclic_specs = [(3, 24.0), (2, 12.0), (5, 7.0), (0, 4.0)]

        for idx, period in cyclic_specs:
            if idx < n_features:
                x_col = X[:, idx:idx+1]
                for k in range(1, self.n_harmonics + 1):
                    angle = 2.0 * np.pi * k * x_col / period
                    features.append(np.sin(angle))
                    features.append(np.cos(angle))

        # 2. Weather non-linear polynomial features (temp, hum, windspeed)
        # Weather indices: 8 (temp), 9 (atemp), 10 (hum), 11 (windspeed)
        weather_indices = [idx for idx in [8, 9, 10, 11] if idx < n_features]
        if weather_indices:
            weather_X = X[:, weather_indices]
            for deg in range(2, self.poly_degree + 1):
                features.append(weather_X ** deg)

        # 3. Domain Interactions: (hr * workingday), (temp * hum), (temp * windspeed)
        if n_features >= 12:
            # hr (3) * workingday (6)
            features.append((X[:, 3:4] * X[:, 6:7]))
            # temp (8) * hum (10)
            features.append((X[:, 8:9] * X[:, 10:11]))
            # temp (8) * windspeed (11)
            features.append((X[:, 8:9] * X[:, 11:12]))

        return np.hstack(features)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
