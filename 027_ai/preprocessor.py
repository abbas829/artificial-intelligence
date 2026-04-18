"""
Data Preprocessor Module
Generated in Day 27 Exercise
"""
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class DataPreprocessor:
    def __init__(self, poly_degree=2):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)

    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        return self.poly.fit_transform(X_scaled)

if __name__ == "__main__":
    import numpy as np
    X = np.random.randn(100, 5)
    prep = DataPreprocessor()
    print(f"Original: {X.shape}, Processed: {prep.fit_transform(X).shape}")
