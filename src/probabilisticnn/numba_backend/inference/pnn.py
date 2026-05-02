import numpy as np

from numba_backend.kernels import resolve_kernel

def pnn_jit_inference(
    kernel: str,
    X: np.ndarray,
    W: np.ndarray,
    y_encoded: np.ndarray,
    n_classes: int,
    likelihood_multiplier: np.ndarray,
    bandwidth,
    bandwidth_sharing: str,
    normalized: bool = False,
):
    """Compute PNN output with the numba kernel backend."""
    likelihood_multiplier = np.asarray(likelihood_multiplier, dtype=X.dtype)
    bandwidth = np.asarray(bandwidth, dtype=X.dtype)
    kernel_ = resolve_kernel(kernel)
    K = kernel_(X, W, bandwidth, bandwidth_sharing, normalized)

    class_mask = np.zeros((y_encoded.shape[0], n_classes), dtype=X.dtype)
    class_mask[np.arange(y_encoded.shape[0]), y_encoded] = 1.0

    f = np.dot(K, class_mask) / n_classes
    posterior = f * likelihood_multiplier
    return np.argmax(posterior, axis=1)
