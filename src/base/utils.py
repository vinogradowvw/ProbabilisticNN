import numpy as np


def normalize_l2(X: np.ndarray) -> np.ndarray:
    """Normalize each sample to unit L2 norm."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return X / norms