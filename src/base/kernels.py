import numpy as np
from base.types import KernelCallable


def _validate_bandwidth_numpy(bandwidth) -> None:
    if np.any(np.asarray(bandwidth) <= 0):
        raise ValueError("`bandwidth` must be strictly positive.")


def gaussian_kernel(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth,  # float or array-like with shape (n_features,), (n_patterns, 1) or (n_patterns, n_features)
    normalized: bool = False,
) -> np.ndarray:
    """Compute Gaussian kernel values between samples in X and W."""
    _validate_bandwidth_numpy(bandwidth)

    bandwidth_sq = np.square(np.asarray(bandwidth))
    if normalized and np.ndim(bandwidth_sq) == 0:  # bandwidth is a scalar
        similarities = np.matmul(X, W.T)
        return np.exp((similarities - 1.0) / bandwidth_sq)

    if np.ndim(bandwidth_sq) == 0:  # bandwidth is a scalar and normalized is False
        x_norm_sq = np.square(X).sum(axis=1, keepdims=True)  # (batch_size, 1)
        w_norm_sq = np.square(W).sum(axis=1) # (n_patterns,)
        l2_norm_sq = x_norm_sq + w_norm_sq - 2.0 * np.matmul(X, W.T)
        return np.exp(-(l2_norm_sq / (2.0 * bandwidth_sq)))
    
    if np.ndim(bandwidth_sq) == 1:
        if bandwidth_sq.shape[0] == W.shape[1]:  # bandwidth shape is (n_features,)
            bandw_inv = 1 / (2 * bandwidth_sq) # (n_features,)
            x_norm_sq = (
                (np.square(X) * bandw_inv)  # per-feature bandwidth normalization
                .sum(axis=1, keepdims=True)  # (batch_size, 1)
            )
            w_norm_sq = (
                (np.square(W) * bandw_inv)  # per-feature bandwidth normalization
                .sum(axis=1)  # (n_patterns,)
            )
            return (
                np.exp(
                    -(x_norm_sq + w_norm_sq - 2.0 * np.matmul(X, (W * bandw_inv).T))
                )
            )
        else:
            raise ValueError(
                "Invadid bandwidth shape. Expected (n_features,), (n_patterns,) or (n_patterns, n_features), got {}".format(bandwidth_sq.shape)
            )
    
    elif np.ndim(bandwidth_sq) == 2:
        if bandwidth_sq.shape[0] == W.shape[0] and bandwidth_sq.shape[1] == W.shape[1]:  # bandwidth shape is (n_patterns, n_features)
            bandw_inv = 1 / (2 * bandwidth_sq) # (n_patterns, n_features)
            x_norm_sq = np.matmul(np.square(X), bandw_inv.T)  # (batch_size, n_patterns)
            w_norm_sq = (np.square(W) * bandw_inv).sum(axis=1) # (n_patterns,)
            l2_norm_sq = x_norm_sq + w_norm_sq - 2.0 * np.matmul(X, (W * bandw_inv).T)  # (batch_size, n_patterns)
            return np.exp(-(l2_norm_sq))
        elif bandwidth_sq.shape[0] == W.shape[0] and bandwidth_sq.shape[1] == 1:  # bandwidth shape is (n_patterns, 1) per-class
            bandw_inv = 1 / (2 * bandwidth_sq.squeeze()) # (n_patterns,)
            x_norm_sq = np.square(X).sum(axis=1, keepdims=True)  # (batch_size, 1)
            w_norm_sq = np.square(W).sum(axis=1) # (n_patterns,)
            l2_norm_sq = x_norm_sq + w_norm_sq - 2.0 * np.matmul(X, W.T)  # (batch_size, n_patterns)
            return np.exp(-(l2_norm_sq * bandw_inv))
        else:
            raise ValueError(
                "Invadid bandwidth shape. Expected (n_features,), (n_patterns,) or (n_patterns, n_features), got {}".format(bandwidth_sq.shape)
            )
    else:
        raise ValueError(
            "Invadid bandwidth shape. Expected (n_features,), (n_patterns,) or (n_patterns, n_features), got {}".format(bandwidth_sq.shape)
        )

KERNEL_REGISTRY: dict[str, KernelCallable] = {
    "gaussian": gaussian_kernel,
}

def __resolve_kernel(kernel: str):
    try:
        return KERNEL_REGISTRY[kernel.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(KERNEL_REGISTRY))
        raise ValueError(f"Unknown kernel={kernel!r}. Available: {available}") from exc
