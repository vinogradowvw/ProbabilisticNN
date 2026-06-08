import math

import numpy as np
from probabilisticnn.base.types import KernelCallable
from numba import njit


@njit(fastmath=True, cache=True)
def _rowwise_product(values: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=values.dtype)
    for i in range(values.shape[0]):
        out[i] = np.prod(values[i])
    return out


# ==============================================================================
# Matrix-level kernels  (return full K matrix)
# Used by the public gaussian_kernel / laplacian_kernel / exponential_kernel API.
# ==============================================================================

# ------------------------------------------------------------------------------
# gaussian kernel
# Uses the : ||x-w||^2 = ||x||^2 + ||w||^2 - 2 x·w — no 3-D tensor.
# ------------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _gaussian_kernel_scalar_bandw(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.float32 | np.float64,
    normalized: bool = False,
) -> np.ndarray:
    bandwidth_sq = bandwidth * bandwidth
    if normalized:
        similarities = np.dot(X, W.T)
        one = similarities.dtype.type(1)
        zero = similarities.dtype.type(0)
        scaled_distance = (one - similarities) / bandwidth_sq
        scaled_distance = np.maximum(scaled_distance, zero)
        return np.exp(-scaled_distance)

    x_norm_sq = np.square(X).sum(axis=1).reshape((X.shape[0], 1))
    w_norm_sq = np.square(W).sum(axis=1)
    cross_term = np.dot(X, W.T)
    l2_norm_sq = x_norm_sq + w_norm_sq - (cross_term + cross_term)
    scaled_distance = l2_norm_sq / (bandwidth_sq + bandwidth_sq)
    scaled_distance = np.maximum(scaled_distance, scaled_distance.dtype.type(0))
    return np.exp(-scaled_distance)


@njit(parallel=True, fastmath=True, cache=True)
def _gaussian_kernel_per_feature(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    bandwidth_sq = np.square(bandwidth)
    bandw_inv = np.reciprocal(bandwidth_sq + bandwidth_sq)
    x_norm_sq = (
        (np.square(X) * bandw_inv)
        .sum(axis=1)
        .reshape((X.shape[0], 1))
    )
    w_norm_sq = (
        (np.square(W) * bandw_inv)
        .sum(axis=1)
    )
    cross_term = np.dot(X, (W * bandw_inv).T)
    scaled_distance = x_norm_sq + w_norm_sq - (cross_term + cross_term)
    scaled_distance = np.maximum(scaled_distance, scaled_distance.dtype.type(0))
    return np.exp(-scaled_distance)


@njit(parallel=True, fastmath=True, cache=True)
def _gaussian_kernel_per_class(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    dtype = X.dtype
    n_features = X.shape[1]
    bandwidth_sq = np.square(bandwidth)
    bandw_inv = np.reciprocal(bandwidth_sq + bandwidth_sq)
    x_norm_sq = np.square(X).sum(axis=1).reshape((X.shape[0], 1))
    w_norm_sq = np.square(W).sum(axis=1)
    cross_term = np.dot(X, W.T)
    scaled_distance = (x_norm_sq + w_norm_sq - (cross_term + cross_term)) * bandw_inv
    scaled_distance = np.maximum(scaled_distance, scaled_distance.dtype.type(0))
    normalization = dtype.type(np.power(2.0 * np.pi, -0.5 * n_features)) * np.power(bandwidth, -n_features)
    normalization = normalization.astype(dtype)
    return np.exp(-scaled_distance) * normalization.reshape((1, normalization.shape[0]))


@njit(parallel=True, fastmath=True, cache=True)
def _gaussian_kernel_per_class_per_feature(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    dtype = X.dtype
    n_features = X.shape[1]
    bandwidth_sq = np.square(bandwidth)
    bandw_inv = np.reciprocal(bandwidth_sq + bandwidth_sq)
    x_norm_sq = np.dot(np.square(X), bandw_inv.T)
    w_norm_sq = (np.square(W) * bandw_inv).sum(axis=1)
    cross_term = np.dot(X, (W * bandw_inv).T)
    scaled_distance = x_norm_sq + w_norm_sq - (cross_term + cross_term)
    scaled_distance = np.maximum(scaled_distance, scaled_distance.dtype.type(0))
    normalization = dtype.type(np.power(2.0 * np.pi, -0.5 * n_features)) / _rowwise_product(bandwidth)
    normalization = normalization.astype(dtype)
    return np.exp(-scaled_distance) * normalization.reshape((1, normalization.shape[0]))


# ------------------------------------------------------------------------------
# laplacian kernel
# L1 distance cannot use the factorization — explicit loops avoid the
# (n_samples, n_patterns, n_features) intermediate tensor.
# ------------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def _laplacian_kernel_scalar_bandw(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.float32 | np.float64,
    normalized: bool = False,
) -> np.ndarray:
    n_samples = X.shape[0]
    n_patterns = W.shape[0]
    n_features = X.shape[1]
    dtype = X.dtype
    out = np.empty((n_samples, n_patterns), dtype=dtype)
    inv_bw = dtype.type(1.0) / bandwidth
    for i in range(n_samples):
        for j in range(n_patterns):
            acc = dtype.type(0.0)
            for k in range(n_features):
                d = X[i, k] - W[j, k]
                if d < dtype.type(0.0):
                    d = -d
                acc += d
            out[i, j] = dtype.type(math.exp(float(-acc * inv_bw)))
    return out


@njit(fastmath=True, cache=True)
def _laplacian_kernel_per_feature(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    n_samples = X.shape[0]
    n_patterns = W.shape[0]
    n_features = X.shape[1]
    dtype = X.dtype
    out = np.empty((n_samples, n_patterns), dtype=dtype)
    for i in range(n_samples):
        for j in range(n_patterns):
            acc = dtype.type(0.0)
            for k in range(n_features):
                d = X[i, k] - W[j, k]
                if d < dtype.type(0.0):
                    d = -d
                acc += d / bandwidth[k]
            out[i, j] = dtype.type(math.exp(float(-acc)))
    return out


@njit(fastmath=True, cache=True)
def _laplacian_kernel_per_class(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    dtype = X.dtype
    n_samples = X.shape[0]
    n_patterns = W.shape[0]
    n_features = X.shape[1]
    out = np.empty((n_samples, n_patterns), dtype=dtype)
    base_c = dtype.type(np.power(2.0, -n_features))
    norm = (base_c * np.power(bandwidth, -n_features)).astype(dtype)
    for i in range(n_samples):
        for j in range(n_patterns):
            acc = dtype.type(0.0)
            for k in range(n_features):
                d = X[i, k] - W[j, k]
                if d < dtype.type(0.0):
                    d = -d
                acc += d
            out[i, j] = dtype.type(math.exp(float(-acc / bandwidth[j]))) * norm[j]
    return out


@njit(fastmath=True, cache=True)
def _laplacian_kernel_per_class_per_feature(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    dtype = X.dtype
    n_samples = X.shape[0]
    n_patterns = W.shape[0]
    n_features = X.shape[1]
    out = np.empty((n_samples, n_patterns), dtype=dtype)
    base_c = dtype.type(np.power(2.0, -n_features))
    norm = (base_c / _rowwise_product(bandwidth.astype(dtype))).astype(dtype)
    for i in range(n_samples):
        for j in range(n_patterns):
            acc = dtype.type(0.0)
            for k in range(n_features):
                d = X[i, k] - W[j, k]
                if d < dtype.type(0.0):
                    d = -d
                acc += d / bandwidth[j, k]
            out[i, j] = dtype.type(math.exp(float(-acc))) * norm[j]
    return out


# ------------------------------------------------------------------------------
# exponential kernel — uses kernel trick for L2, no 3-D tensor needed.
# ------------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _exponential_kernel_scalar_bandw(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.float32 | np.float64,
    normalized: bool = False,
) -> np.ndarray:
    if normalized:
        similarities = np.dot(X, W.T)
        one = similarities.dtype.type(1)
        zero = similarities.dtype.type(0)
        l2_norm_sq = one - similarities
        l2_norm_sq = l2_norm_sq + l2_norm_sq
        l2_norm_sq = np.maximum(l2_norm_sq, zero)
        l2_norm = np.sqrt(l2_norm_sq)
        scaled_distance = l2_norm / bandwidth
        scaled_distance = np.maximum(scaled_distance, zero)
        return np.exp(-scaled_distance)

    x_norm_sq = np.square(X).sum(axis=1).reshape((X.shape[0], 1))
    w_norm_sq = np.square(W).sum(axis=1)
    cross_term = np.dot(X, W.T)
    l2_norm_sq = x_norm_sq + w_norm_sq - (cross_term + cross_term)
    l2_norm_sq = np.maximum(l2_norm_sq, l2_norm_sq.dtype.type(0))
    l2_norm = np.sqrt(l2_norm_sq)
    scaled_distance = l2_norm / bandwidth
    scaled_distance = np.maximum(scaled_distance, scaled_distance.dtype.type(0))
    return np.exp(-scaled_distance)


@njit(parallel=True, fastmath=True, cache=True)
def _exponential_kernel_per_feature(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    x_scaled = X / bandwidth
    w_scaled = W / bandwidth
    x_norm_sq = np.square(x_scaled).sum(axis=1).reshape((X.shape[0], 1))
    w_norm_sq = np.square(w_scaled).sum(axis=1)
    cross_term = np.dot(x_scaled, w_scaled.T)
    scaled_distance_sq = x_norm_sq + w_norm_sq - (cross_term + cross_term)
    scaled_distance_sq = np.maximum(scaled_distance_sq, scaled_distance_sq.dtype.type(0))
    scaled_distance = np.sqrt(scaled_distance_sq)
    scaled_distance = np.maximum(scaled_distance, scaled_distance.dtype.type(0))
    return np.exp(-scaled_distance)


@njit(parallel=True, fastmath=True, cache=True)
def _exponential_kernel_per_class(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    dtype = X.dtype
    n_features = X.shape[1]
    bandw_inv = np.reciprocal(bandwidth)
    bandw_inv_sq = np.square(bandw_inv)
    x_norm_sq = np.square(X).sum(axis=1).reshape((X.shape[0], 1))
    w_norm_sq = np.square(W).sum(axis=1)
    cross_term = np.dot(X, W.T)
    scaled_distance_sq = (x_norm_sq + w_norm_sq - (cross_term + cross_term)) * bandw_inv_sq
    scaled_distance_sq = np.maximum(scaled_distance_sq, scaled_distance_sq.dtype.type(0))
    scaled_distance = np.sqrt(scaled_distance_sq)
    scaled_distance = np.maximum(scaled_distance, scaled_distance.dtype.type(0))
    normalization = dtype.type(
        math.gamma(0.5 * n_features) / (2.0 * np.power(np.pi, 0.5 * n_features) * math.gamma(n_features))
    ) * np.power(bandwidth, -n_features)
    normalization = normalization.astype(dtype)
    return np.exp(-scaled_distance) * normalization.reshape((1, normalization.shape[0]))


@njit(parallel=True, fastmath=True, cache=True)
def _exponential_kernel_per_class_per_feature(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    dtype = X.dtype
    n_features = X.shape[1]
    bandw_inv = np.reciprocal(bandwidth)
    bandw_inv_sq = np.square(bandw_inv)
    x_norm_sq = np.dot(np.square(X), bandw_inv_sq.T)
    w_norm_sq = (np.square(W) * bandw_inv_sq).sum(axis=1)
    cross_term = np.dot(X, (W * bandw_inv_sq).T)
    scaled_distance_sq = x_norm_sq + w_norm_sq - (cross_term + cross_term)
    scaled_distance_sq = np.maximum(scaled_distance_sq, scaled_distance_sq.dtype.type(0))
    scaled_distance = np.sqrt(scaled_distance_sq)
    scaled_distance = np.maximum(scaled_distance, scaled_distance.dtype.type(0))
    normalization = dtype.type(
        math.gamma(0.5 * n_features) / (2.0 * np.power(np.pi, 0.5 * n_features) * math.gamma(n_features))
    ) / _rowwise_product(bandwidth)
    normalization = normalization.astype(dtype)
    return np.exp(-scaled_distance) * normalization.reshape((1, normalization.shape[0]))


# ==============================================================================
# Matrix-kernel dispatch
# ==============================================================================

# Maps (kernel, bandwidth_sharing) to matrix-level @njit(fastmath=True) function.
_MATRIX_KERNEL_REGISTRY: dict = {
    ("gaussian", "scalar"):  _gaussian_kernel_scalar_bandw,
    ("gaussian", "per_feature"): _gaussian_kernel_per_feature,
    ("gaussian", "per_class"): _gaussian_kernel_per_class,
    ("gaussian", "per_class_per_feature"): _gaussian_kernel_per_class_per_feature,

    ("laplacian", "scalar"): _laplacian_kernel_scalar_bandw,
    ("laplacian", "per_feature"): _laplacian_kernel_per_feature,
    ("laplacian", "per_class"):  _laplacian_kernel_per_class,
    ("laplacian", "per_class_per_feature"): _laplacian_kernel_per_class_per_feature,

    ("exponential", "scalar"): _exponential_kernel_scalar_bandw,
    ("exponential", "per_feature"): _exponential_kernel_per_feature,
    ("exponential", "per_class"): _exponential_kernel_per_class,
    ("exponential", "per_class_per_feature"): _exponential_kernel_per_class_per_feature,
}


def resolve_matrix_kernel(kernel: str, bandwidth_sharing: str):
    """Return the matrix-level @njit kernel for (kernel, bandwidth_sharing)."""
    key = (kernel.lower(), bandwidth_sharing)
    try:
        return _MATRIX_KERNEL_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"No matrix kernel for kernel={kernel!r}, bandwidth_sharing={bandwidth_sharing!r}"
        ) from exc


# ==============================================================================
# Public matrix-kernel API  (kept for direct use / testing)
# ==============================================================================

def gaussian_kernel(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth,
    bandwidth_sharing: str | None = None,
    normalized: bool = False,
) -> np.ndarray:
    """Compute Gaussian kernel values between samples in X and W."""
    if np.any(bandwidth <= 0):
        raise ValueError("`bandwidth` must be strictly positive.")

    dtype = X.dtype
    X = np.asarray(X, dtype=dtype)
    W = np.asarray(W, dtype=dtype)
    bandwidth_arr = np.asarray(bandwidth, dtype=dtype)

    if bandwidth_sharing == "scalar":
        return _gaussian_kernel_scalar_bandw(X, W, bandwidth_arr[()], normalized)
    elif bandwidth_sharing == "per_feature":
        return _gaussian_kernel_per_feature(X, W, bandwidth_arr, normalized)
    elif bandwidth_sharing == "per_class":
        return _gaussian_kernel_per_class(X, W, bandwidth_arr, normalized)
    elif bandwidth_sharing == "per_class_per_feature":
        return _gaussian_kernel_per_class_per_feature(X, W, bandwidth_arr, normalized)
    else:
        raise ValueError(f"Unknown bandwidth_sharing.")


def laplacian_kernel(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth,
    bandwidth_sharing: str | None = None,
    normalized: bool = False,
) -> np.ndarray:
    """Compute Laplacian kernel values between samples in X and W."""
    if np.any(bandwidth <= 0):
        raise ValueError("`bandwidth` must be strictly positive.")

    dtype = X.dtype
    X = np.asarray(X, dtype=dtype)
    W = np.asarray(W, dtype=dtype)
    bandwidth_arr = np.asarray(bandwidth, dtype=dtype)

    if bandwidth_sharing == "scalar":
        return _laplacian_kernel_scalar_bandw(X, W, bandwidth_arr[()], normalized)
    elif bandwidth_sharing == "per_feature":
        return _laplacian_kernel_per_feature(X, W, bandwidth_arr, normalized)
    elif bandwidth_sharing == "per_class":
        return _laplacian_kernel_per_class(X, W, bandwidth_arr, normalized)
    elif bandwidth_sharing == "per_class_per_feature":
        return _laplacian_kernel_per_class_per_feature(X, W, bandwidth_arr, normalized)
    else:
        raise ValueError(f"Unknown bandwidth_sharing")


def exponential_kernel(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth,
    bandwidth_sharing: str | None = None,
    normalized: bool = False,
) -> np.ndarray:
    """Compute Exponential kernel values between samples in X and W."""
    if np.any(bandwidth <= 0):
        raise ValueError("`bandwidth` must be strictly positive.")

    dtype = X.dtype
    X = np.asarray(X, dtype=dtype)
    W = np.asarray(W, dtype=dtype)
    bandwidth_arr = np.asarray(bandwidth, dtype=dtype)

    if bandwidth_sharing == "scalar":
        return _exponential_kernel_scalar_bandw(X, W, bandwidth_arr[()], normalized)
    elif bandwidth_sharing == "per_feature":
        return _exponential_kernel_per_feature(X, W, bandwidth_arr, normalized)
    elif bandwidth_sharing == "per_class":
        return _exponential_kernel_per_class(X, W, bandwidth_arr, normalized)
    elif bandwidth_sharing == "per_class_per_feature":
        return _exponential_kernel_per_class_per_feature(X, W, bandwidth_arr, normalized)
    else:
        raise ValueError(f"Unknown bandwidth_sharing.")


KERNEL_REGISTRY: dict[str, KernelCallable] = {
    "gaussian": gaussian_kernel,
    "laplacian": laplacian_kernel,
    "exponential": exponential_kernel,
}


def resolve_kernel(kernel: str):
    try:
        return KERNEL_REGISTRY[kernel.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(KERNEL_REGISTRY))
        raise ValueError(f"Unknown kernel={kernel!r}. Available: {available}") from exc
