"""
Data Preprocessing Pipeline
Version: 1.0.0
Author: AI Learning Path
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Transformer that clips outliers to specified percentiles.

    Parameters:
    -----------
    lower_percentile : float
        Lower bound percentile (0-100)
    upper_percentile : float
        Upper bound percentile (0-100)
    """

    def __init__(self, lower_percentile=1.0, upper_percentile=99.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

def create_polynomial_features(X, degree=2):
    """Generate polynomial features up to specified degree."""
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)

if __name__ == "__main__":
    # Demo
    data = np.random.randn(100, 3)
    clipper = OutlierClipper()
    clipped = clipper.fit_transform(data)
    print(f"Original range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Clipped range: [{clipped.min():.2f}, {clipped.max():.2f}]")
