import numpy as np
import torch
from base.types import KernelCallable


def _validate_bandwidth_numpy(bandwidth) -> None:
    if np.any(np.asarray(bandwidth) <= 0):
        raise ValueError("`bandwidth` must be strictly positive.")


def _validate_bandwidth_torch(bandwidth) -> None:
    if isinstance(bandwidth, torch.Tensor):
        if torch.any(bandwidth <= 0):
            raise ValueError("`bandwidth` must be strictly positive.")
        return

    if bandwidth <= 0:
        raise ValueError("`bandwidth` must be strictly positive.")


def gaussian_kernel(
    X: np.ndarray,
    W: np.ndarray,
    bandwidth,  # float or array-like with shape (batch_size, n_patterns, n_features)
    normalized: bool = False,
) -> np.ndarray:
    """Compute Gaussian kernel values between samples in X and W."""
    _validate_bandwidth_numpy(bandwidth)

    bandwidth_sq = bandwidth ** 2
    if normalized and np.ndim(bandwidth_sq) == 0:  # bandwidth is a scalar
        similarities = np.matmul(X, W.T)
        return np.exp((similarities - 1.0) / bandwidth_sq)

    if np.ndim(bandwidth_sq) == 0:  # bandwidth is a scalar and normalized is False
        x_norm_sq = np.square(X).sum(axis=1, keepdims=True)  # (batch_size, 1)
        w_norm_sq = np.square(W).sum(axis=1) # (n_patterns,)
        l2_norm_sq = x_norm_sq + w_norm_sq - 2.0 * np.matmul(X, W.T)
        return np.exp(-(l2_norm_sq / (2.0 * bandwidth_sq)))

    # bandwidth is an array-like with shape (batch_size, n_patterns, n_features)
    squared_distances = (X[:, np.newaxis, :] - W[np.newaxis, :, :]) ** 2
    scaled_distances = np.sum(squared_distances / (2.0 * bandwidth_sq), axis=2)
    return np.exp(-scaled_distances)


def gaussian_kernel_t(
    X: torch.Tensor,
    W: torch.Tensor,
    bandwidth,  # float or array-like with shape (batch_size, n_patterns, n_features)
    normalized: bool = False,
) -> torch.Tensor:
    """Compute Gaussian kernel values between samples in X and W. (pytorch)"""
    _validate_bandwidth_torch(bandwidth)

    bandwidth_sq = bandwidth ** 2
    if normalized and (not isinstance(bandwidth_sq, torch.Tensor) or bandwidth_sq.ndim == 0):
        # bandwidth is a scalar
        similarities = torch.matmul(X, W.T)
        return torch.exp((similarities - 1.0) / bandwidth_sq)

    if (not isinstance(bandwidth_sq, torch.Tensor) or bandwidth_sq.ndim == 0):  # bandwidth is a scalar and normalized is False
        x_norm_sq = torch.square(X).sum(dim=1, keepdim=True)  # (batch_size, 1)
        w_norm_sq = torch.square(W).sum(dim=1) # (n_patterns,)
        l2_norm_sq = x_norm_sq + w_norm_sq - 2.0 * torch.matmul(X, W.T)
        return torch.exp(-(l2_norm_sq / (2.0 * bandwidth_sq)))

    # bandwidth is an array-like with shape (batch_size, n_patterns, n_features)
    squared_distances = (X[:, None, :] - W[None, :, :]) ** 2
    scaled_distances = torch.sum(squared_distances / (2.0 * bandwidth_sq), dim=2)
    return torch.exp(-scaled_distances)


KERNEL_REGISTRY: dict[str, KernelCallable] = {
    "gaussian": gaussian_kernel,
}

KERNEL_T_REGISTRY = {
    "gaussian": gaussian_kernel_t,
}


def __resolve_kernel(kernel: str, torch: bool = False):
    try:
        if torch:
            return KERNEL_T_REGISTRY[kernel.lower()]
        else:
            return KERNEL_REGISTRY[kernel.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(KERNEL_T_REGISTRY if torch else KERNEL_REGISTRY))
        raise ValueError(f"Unknown kernel={kernel!r}. Available: {available}") from exc
