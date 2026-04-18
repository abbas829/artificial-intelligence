
"""
ML Utility Functions
Production-ready helper functions for data preprocessing.
"""
import numpy as np
from typing import Tuple, Optional

def robust_scale(X: np.ndarray, quantile_range: Tuple[float, float] = (25.0, 75.0)) -> np.ndarray:
    """
    Robust scaling using median and IQR (interquartile range).

    More robust to outliers than standard z-score normalization.
    """
    median = np.median(X, axis=0)
    q1 = np.percentile(X, quantile_range[0], axis=0)
    q3 = np.percentile(X, quantile_range[1], axis=0)
    iqr = q3 - q1
    return (X - median) / (iqr + 1e-8)

def train_val_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    val_size: float = 0.1, 
    test_size: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple:
    """
    Split data into train/validation/test sets.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if random_state:
        np.random.seed(random_state)

    n = len(X)
    indices = np.random.permutation(n)

    test_end = int(n * test_size)
    val_end = test_end + int(n * val_size)

    test_idx = indices[:test_end]
    val_idx = indices[test_end:val_end]
    train_idx = indices[val_end:]

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx]
    )

print("✅ ML utilities module loaded successfully!")
