import numpy as np

from probabilisticnn.numba_backend.kernels import resolve_matrix_kernel


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
    """Compute PNN output with the numba kernel backend.

    Mirrors the numpy path (compute K, then reduce with BLAS @) for a fair
    apples-to-apples comparison. Faster because fastmath enables FP rewrites
    and JIT eliminates Python overhead.
    """
    likelihood_multiplier = np.asarray(likelihood_multiplier, dtype=X.dtype)
    bandwidth_arr = np.asarray(bandwidth, dtype=X.dtype)
    bandwidth_arg = bandwidth_arr[()] if bandwidth_sharing == "scalar" else bandwidth_arr

    n_patterns = W.shape[0]
    class_mask = np.zeros((n_patterns, n_classes), dtype=X.dtype)
    class_mask[np.arange(n_patterns), y_encoded] = 1.0

    W_cast = np.asarray(W, dtype=X.dtype)
    K = resolve_matrix_kernel(kernel, bandwidth_sharing)(X, W_cast, bandwidth_arg, normalized)
    f = K @ class_mask
    return np.argmax(f * likelihood_multiplier, axis=1).astype(np.int64)
