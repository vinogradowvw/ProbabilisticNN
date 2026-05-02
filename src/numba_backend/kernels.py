import numpy as np
from base.types import KernelCallable
from numba import njit


# ------------------------------------------------------------------------------
# jitted kernels for each bandwidth_sharing
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# gaussian kernel
# ------------------------------------------------------------------------------
@njit
def _gaussian_kernel_scalar_bandw(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.float32 | np.float64,
    normalized: bool = False,
) -> np.ndarray:
    bandwidth_sq = bandwidth * bandwidth
    if normalized:
        similarities = np.dot(X, W.T)
        scaled_distance = (np.ones_like(similarities) - similarities) / bandwidth_sq
        scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
        return np.exp(-scaled_distance)

    x_norm_sq = np.square(X).sum(axis=1).reshape((X.shape[0], 1))
    w_norm_sq = np.square(W).sum(axis=1)
    cross_term = np.dot(X, W.T)
    l2_norm_sq = x_norm_sq + w_norm_sq - (cross_term + cross_term)
    scaled_distance = l2_norm_sq / (bandwidth_sq + bandwidth_sq)
    scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
    return np.exp(-scaled_distance)


@njit
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
    scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
    return np.exp(-scaled_distance)


@njit
def _gaussian_kernel_per_class(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    bandwidth_sq = np.square(bandwidth)
    bandw_inv = np.reciprocal(bandwidth_sq + bandwidth_sq)
    x_norm_sq = np.square(X).sum(axis=1).reshape((X.shape[0], 1))
    w_norm_sq = np.square(W).sum(axis=1)
    cross_term = np.dot(X, W.T)
    scaled_distance = (x_norm_sq + w_norm_sq - (cross_term + cross_term)) * bandw_inv
    scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
    return np.exp(-scaled_distance)


@njit
def _gaussian_kernel_per_class_per_feature(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    bandwidth_sq = np.square(bandwidth)
    bandw_inv = np.reciprocal(bandwidth_sq + bandwidth_sq)
    x_norm_sq = np.dot(np.square(X), bandw_inv.T)
    w_norm_sq = (np.square(W) * bandw_inv).sum(axis=1)
    cross_term = np.dot(X, (W * bandw_inv).T)
    scaled_distance = x_norm_sq + w_norm_sq - (cross_term + cross_term)
    scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
    return np.exp(-scaled_distance)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# laplacian kernel  
# ------------------------------------------------------------------------------
@njit
def _laplacian_kernel_scalar_bandw(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.float32 | np.float64,
    normalized: bool = False,
) -> np.ndarray:
    l1_norm = np.abs(X[:, None, :] - W[None, :, :]).sum(axis=2)
    scaled_distance = l1_norm / bandwidth
    scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
    return np.exp(-scaled_distance)


@njit
def _laplacian_kernel_per_feature(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    l1_norm_normalized = (
        np.abs(X[:, None, :] / bandwidth - W[None, :, :] / bandwidth)
        .sum(axis=2)
    )
    l1_norm_normalized = np.maximum(l1_norm_normalized, np.zeros_like(l1_norm_normalized))
    return np.exp(-l1_norm_normalized)


@njit
def _laplacian_kernel_per_class(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    l1_norm = np.abs(X[:, None, :] - W[None, :, :]).sum(axis=2)
    scaled_distance = l1_norm / bandwidth
    scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
    return np.exp(-scaled_distance)


@njit
def _laplacian_kernel_per_class_per_feature(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    l1_norm_normalized = (
        np.abs((X[:, None, :] - W[None, :, :]) / bandwidth)
        .sum(axis=2)
    )
    l1_norm_normalized = np.maximum(l1_norm_normalized, np.zeros_like(l1_norm_normalized))
    return np.exp(-l1_norm_normalized)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# exponential kernel    
# ------------------------------------------------------------------------------
@njit
def _exponential_kernel_scalar_bandw(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.float32 | np.float64,
    normalized: bool = False,
) -> np.ndarray:
    if normalized:
        similarities = np.dot(X, W.T)
        l2_norm_sq = np.ones_like(similarities) - similarities
        l2_norm_sq = l2_norm_sq + l2_norm_sq
        l2_norm_sq = np.maximum(l2_norm_sq, np.zeros_like(l2_norm_sq))
        l2_norm = np.sqrt(l2_norm_sq)
        scaled_distance = l2_norm / bandwidth
        scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
        return np.exp(-scaled_distance)

    x_norm_sq = np.square(X).sum(axis=1).reshape((X.shape[0], 1))
    w_norm_sq = np.square(W).sum(axis=1)
    cross_term = np.dot(X, W.T)
    l2_norm_sq = x_norm_sq + w_norm_sq - (cross_term + cross_term)
    l2_norm_sq = np.maximum(l2_norm_sq, np.zeros_like(l2_norm_sq))
    l2_norm = np.sqrt(l2_norm_sq)
    scaled_distance = l2_norm / bandwidth
    scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
    return np.exp(-scaled_distance)


@njit
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
    scaled_distance_sq = np.maximum(scaled_distance_sq, np.zeros_like(scaled_distance_sq))
    scaled_distance = np.sqrt(scaled_distance_sq)
    scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
    return np.exp(-scaled_distance)


@njit
def _exponential_kernel_per_class(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    bandw_inv = np.reciprocal(bandwidth)
    bandw_inv_sq = np.square(bandw_inv)
    x_norm_sq = np.square(X).sum(axis=1).reshape((X.shape[0], 1))
    w_norm_sq = np.square(W).sum(axis=1)
    cross_term = np.dot(X, W.T)
    scaled_distance_sq = (x_norm_sq + w_norm_sq - (cross_term + cross_term)) * bandw_inv_sq
    scaled_distance_sq = np.maximum(scaled_distance_sq, np.zeros_like(scaled_distance_sq))
    scaled_distance = np.sqrt(scaled_distance_sq)
    scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
    return np.exp(-scaled_distance)


@njit
def _exponential_kernel_per_class_per_feature(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth: np.ndarray,
    normalized: bool = False,
) -> np.ndarray:
    bandw_inv = np.reciprocal(bandwidth)
    bandw_inv_sq = np.square(bandw_inv)
    x_norm_sq = np.dot(np.square(X), bandw_inv_sq.T)
    w_norm_sq = (np.square(W) * bandw_inv_sq).sum(axis=1)
    cross_term = np.dot(X, (W * bandw_inv_sq).T)
    scaled_distance_sq = x_norm_sq + w_norm_sq - (cross_term + cross_term)
    scaled_distance_sq = np.maximum(scaled_distance_sq, np.zeros_like(scaled_distance_sq))
    scaled_distance = np.sqrt(scaled_distance_sq)
    scaled_distance = np.maximum(scaled_distance, np.zeros_like(scaled_distance))
    return np.exp(-scaled_distance)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# higer level API for kernels
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# gaussian kernel
# ------------------------------------------------------------------------------
def gaussian_kernel(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth,  # float or array-like with shape (n_features,), (n_patterns,) or (n_patterns, n_features)
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
    bandwidth,  # float or array-like with shape (n_features,), (n_patterns,) or (n_patterns, n_features)
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
    bandwidth,  # float or array-like with shape (n_features,), (n_patterns,) or (n_patterns, n_features)
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
