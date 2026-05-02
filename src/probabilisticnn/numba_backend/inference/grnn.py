import numpy as np

from probabilisticnn.numba_backend.kernels import resolve_kernel


def grnn_jit_inference(
    kernel: str,
    X: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    bandwidth,
    bandwidth_sharing: str,
    normalized: bool = False,
):
    """Compute GRNN output with a single full kernel evaluation."""
    kernel_ = resolve_kernel(kernel)
    bandwidth = np.asarray(bandwidth, dtype=X.dtype)
    K = kernel_(X, W, bandwidth, bandwidth_sharing, normalized)
    denom = np.sum(K, axis=1)
    nom = np.dot(K, y)
    return np.divide(
        nom,
        denom,
        out=np.zeros_like(nom, dtype=X.dtype),
        where=denom > 0,
    )
