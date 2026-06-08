import numpy as np

from probabilisticnn.numba_backend.kernels import resolve_matrix_kernel


def grnn_jit_inference(
    kernel: str,
    X: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    bandwidth,
    bandwidth_sharing: str,
    normalized: bool = False,
):
    """Compute GRNN output with the numba kernel backend.

    Mirrors the numpy path (compute K, then reduce with BLAS @) for a fair
    apples-to-apples comparison. Faster because fastmath enables FP rewrites
    and JIT eliminates Python overhead.
    """
    bandwidth_arr = np.asarray(bandwidth, dtype=X.dtype)
    bandwidth_arg = bandwidth_arr[()] if bandwidth_sharing == "scalar" else bandwidth_arr

    W_cast = np.asarray(W, dtype=X.dtype)
    K = resolve_matrix_kernel(kernel, bandwidth_sharing)(X, W_cast, bandwidth_arg, normalized)
    nom = K @ y
    denom = K.sum(axis=1)
    safe_denom = np.where(denom > 0, denom, np.ones_like(denom))
    return np.where(denom > 0, nom / safe_denom, np.zeros_like(nom))
