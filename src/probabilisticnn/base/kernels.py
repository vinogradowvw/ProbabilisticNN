import numpy as np
from probabilisticnn.base.types import KernelCallable


# ------------------------------------------------------------------------------
# Utils funcitons
# ------------------------------------------------------------------------------
def _validate_bandwidth_numpy(bandwidth) -> None:
    if np.any(bandwidth <= 0):
        raise ValueError("`bandwidth` must be strictly positive.")


def _exp_from_scaled_distance(scaled_distance):
    """Exponentiate a theoretically non-negative scaled distance safely."""
    scaled_distance = np.asarray(scaled_distance)
    scaled_distance = np.maximum(scaled_distance, scaled_distance.dtype.type(0))
    return np.exp(-scaled_distance)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Kernels
# ------------------------------------------------------------------------------

# gaussian kernel
def gaussian_kernel(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth,  # np.float or array-like with shape (n_features,), (n_patterns,) or (n_patterns, n_features)
    bandwidth_sharing: str,
    normalized: bool = False,
) -> np.ndarray:
    """Compute Gaussian kernel values between samples in X and W."""
    _validate_bandwidth_numpy(bandwidth)

    # скалярный bandwidth
    if bandwidth_sharing == "scalar":
        bandwidth_sq = bandwidth * bandwidth
        if normalized:
            similarities = np.dot(X, W.T)
            scaled_distance = (1.0 - similarities) / bandwidth_sq
            return _exp_from_scaled_distance(scaled_distance)

        x_norm_sq = np.square(X).sum(axis=1, keepdims=True)
        w_norm_sq = np.square(W).sum(axis=1)
        l2_norm_sq = x_norm_sq + w_norm_sq - 2.0 * np.dot(X, W.T)
        scaled_distance = l2_norm_sq / (2.0 * bandwidth_sq)
        return _exp_from_scaled_distance(scaled_distance)

    # bandwidth по каждому признаку
    elif bandwidth_sharing == "per_feature":
        bandwidth_sq = np.square(bandwidth)
        bandw_inv = 1.0 / (2.0 * bandwidth_sq)
        x_norm_sq = (np.square(X) * bandw_inv).sum(axis=1, keepdims=True)
        w_norm_sq = (np.square(W) * bandw_inv).sum(axis=1)
        scaled_distance = x_norm_sq + w_norm_sq - 2.0 * np.dot(X, (W * bandw_inv).T)
        return _exp_from_scaled_distance(scaled_distance)

    # bandwidth по каждому классу
    elif bandwidth_sharing == "per_class":
        bandwidth_sq = np.square(bandwidth)
        bandw_inv = 1.0 / (2.0 * bandwidth_sq)
        x_norm_sq = np.square(X).sum(axis=1, keepdims=True)
        w_norm_sq = np.square(W).sum(axis=1)
        scaled_distance = (x_norm_sq + w_norm_sq - 2.0 * np.dot(X, W.T)) * bandw_inv
        return _exp_from_scaled_distance(scaled_distance)


    # bandwidth по каждому классу по каждому признаку
    elif bandwidth_sharing == "per_class_per_feature":
        bandwidth_sq = np.square(bandwidth)
        bandw_inv = 1.0 / (2.0 * bandwidth_sq)
        x_norm_sq = np.dot(np.square(X), bandw_inv.T)
        w_norm_sq = (np.square(W) * bandw_inv).sum(axis=1)
        scaled_distance = x_norm_sq + w_norm_sq - 2.0 * np.dot(X, (W * bandw_inv).T)
        return _exp_from_scaled_distance(scaled_distance)
    
    else:
        raise ValueError(f"Unknown bandwidth_sharing={bandwidth_sharing!r}.")

    
# laplacian kernel
def laplacian_kernel(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth,  # float or array-like with shape (n_features,), (n_patterns,) or (n_patterns, n_features)
    bandwidth_sharing: str,
    normalized: bool = False,
) -> np.ndarray:
    """Compute Laplacian kernel values between samples in X and W."""
    _validate_bandwidth_numpy(bandwidth)

    if bandwidth_sharing == "scalar":
        l1_norm = np.abs(X[:, None, :] - W[None, :, :]).sum(axis=2)
        scaled_distance = l1_norm / (bandwidth)
        return _exp_from_scaled_distance(scaled_distance)

    elif bandwidth_sharing == "per_feature":
        l1_norm_normalized = (
            np.abs(
                X[:, None, :] / bandwidth  # X shape (n_samples, 1, n_features)  bandwidth shape (n_features,)
                - W[None, :, :] / bandwidth  # W shape (1, n_patterns, n_features)
            )
            .sum(axis=2)  # (n_samples, n_patterns)
        )
        return _exp_from_scaled_distance(l1_norm_normalized)

    elif bandwidth_sharing == "per_class":
        l1_norm = np.abs(X[:, None, :] - W[None, :, :]).sum(axis=2)
        scaled_distance = l1_norm / bandwidth  # bandwidth shape (n_patterns,) - broadcasted
        return _exp_from_scaled_distance(scaled_distance)

    elif bandwidth_sharing == "per_class_per_feature":
        l1_norm_normalized = (
            np.abs(
                (X[:, None, :] - W[None, :, :])
                / bandwidth  # bandwidth shape (n_patterns, n_features)
            ).sum(axis=2)  # (n_samples, n_patterns)
        )
        return _exp_from_scaled_distance(l1_norm_normalized)
    else:
        raise ValueError(f"Unknown bandwidth_sharing={bandwidth_sharing!r}.")

# exponential kernel
def exponential_kernel(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth,  # float or array-like with shape (n_features,), (n_patterns,) or (n_patterns, n_features)
    bandwidth_sharing: str,
    normalized: bool = False,
) -> np.ndarray:
    """Compute exponential kernel values between samples in X and W."""
    _validate_bandwidth_numpy(bandwidth)

    # скалярный bandwidth
    if bandwidth_sharing == "scalar":
        if normalized:
            l2_norm = np.sqrt(np.clip(2*(1-np.dot(X, W.T)), 0, None))
            return _exp_from_scaled_distance(l2_norm / bandwidth)

        x_norm_sq = np.square(X).sum(axis=1, keepdims=True)
        w_norm_sq = np.square(W).sum(axis=1)
        l2_norm_sq = x_norm_sq + w_norm_sq - 2.0 * np.dot(X, W.T)
        l2_norm = np.sqrt(np.clip(l2_norm_sq, 0, None))
        return _exp_from_scaled_distance(l2_norm / bandwidth)

    # bandwidth по каждому признаку
    elif bandwidth_sharing == "per_feature":
        scaled_diff = X[:, None, :] / bandwidth - W[None, :, :] / bandwidth
        scaled_distance = np.sqrt(np.square(scaled_diff).sum(axis=2))
        return _exp_from_scaled_distance(scaled_distance)

    # bandwidth по каждому классу
    elif bandwidth_sharing == "per_class":
        scaled_diff = (X[:, None, :] - W[None, :, :]) / bandwidth[None, :, None]
        scaled_distance = np.sqrt(np.square(scaled_diff).sum(axis=2))
        return _exp_from_scaled_distance(scaled_distance)

    # bandwidth по каждому классу по каждому признаку
    elif bandwidth_sharing == "per_class_per_feature":
        scaled_diff = (X[:, None, :] - W[None, :, :]) / bandwidth[None, :, :]
        scaled_distance = np.sqrt(np.square(scaled_diff).sum(axis=2))
        return _exp_from_scaled_distance(scaled_distance)
    else:
        raise ValueError(f"Unknown bandwidth_sharing={bandwidth_sharing!r}.")


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
