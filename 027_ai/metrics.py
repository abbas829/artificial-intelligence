"""
Classification metrics module.
"""
import numpy as np

def accuracy(y_true, y_pred):
    """Calculate accuracy."""
    return np.mean(np.array(y_true) == np.array(y_pred))

def precision(y_true, y_pred):
    """Calculate precision (positive predictive value)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    """Calculate recall (sensitivity)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

if __name__ == "__main__":
    # Test
    y_t = [1, 0, 1, 1, 0, 1, 0, 0]
    y_p = [1, 0, 1, 0, 0, 1, 1, 0]
    print(f"Accuracy: {accuracy(y_t, y_p):.2f}")
    print(f"Precision: {precision(y_t, y_p):.2f}")
    print(f"Recall: {recall(y_t, y_p):.2f}")
