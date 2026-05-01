import numpy as np


def normalize_l2(X: np.ndarray) -> np.ndarray:
    """Normalize each sample to unit L2 norm."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return X / norms


def cast_to_dtype(X: np.ndarray | np.generic | float | int, dtype: str):
    """Cast arrays or scalars to a specified floating dtype."""

    if dtype == "auto":
        return np.asarray(X)

    if dtype == "float64":
        target_dtype = np.float64
    elif dtype == "float32":
        target_dtype = np.float32
    else:
        raise ValueError(f"Invalid dtype={dtype!r}. Supported: 'float64', 'float32'.")

    if isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=target_dtype)
        if not X.flags.c_contiguous:
            X = np.ascontiguousarray(X)
        return X

    if np.isscalar(X):
        return target_dtype(X)

    raise TypeError("cast_to_dtype supports numpy arrays and scalar numeric values.")


def as_bandwidth_array(bandwidth, dtype=None):
    """Return scalar bandwidths as scalars and vector bandwidths as arrays."""
    arr = np.asarray(bandwidth, dtype=dtype)
    if arr.ndim == 0:
        return arr[()]
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def pattern_block_size(
    n_patterns: int,
    n_features: int,
    bandwidth_sharing: str,
) -> int:
    if n_patterns <= 128:
        return n_patterns

    target_work = 65536

    if bandwidth_sharing == "per_class_per_feature":
        target_work //= 2
    elif bandwidth_sharing == "scalar":
        target_work *= 2

    block = target_work // max(n_features, 1)

    ladder = (64, 128, 256, 512, 1024)
    block = max(64, min(block, 1024))

    block = max(v for v in ladder if v <= block)
    return min(block, n_patterns)


def validate_backend(backend):
    if backend not in {"numpy", "numba"}:
        raise ValueError(f"Unknown backend={backend!r}. Available: 'numpy', 'numba'.")
    if backend == "numba":
        try:
            from numba_backend import pnn_jit_inference, grnn_jit_inference
        except ImportError as exc:
            raise ImportError(
                "Numba backend is not available. Install it with "
                "`pip install probabilisticnn[numba]`. "
                f"Original import error: {exc}"
            ) from exc
