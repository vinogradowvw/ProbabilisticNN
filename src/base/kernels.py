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


def gaussian_kernel_t(
    X: torch.Tensor,
    W: torch.Tensor,
    bandwidth,  # float or array-like with shape (n_features,), (n_patterns, 1) or (n_patterns, n_features)
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

    if bandwidth_sq.ndim == 1:
        if bandwidth_sq.shape[0] == W.shape[1]:  # bandwidth shape is (n_features,)
            bandw_inv = 1 / (2 * bandwidth_sq) # (n_features,)
            x_norm_sq = (
                (torch.square(X) * bandw_inv)
                .sum(dim=1, keepdim=True)
            )
            w_norm_sq = (
                (torch.square(W) * bandw_inv)
                .sum(dim=1)
            )
            return torch.exp(
                -(x_norm_sq + w_norm_sq - 2.0 * torch.matmul(X, (W * bandw_inv).T))
            )
        else:
            raise ValueError(
                "Invadid bandwidth shape. Expected (n_features,), (n_patterns,) or (n_patterns, n_features), got {}".format(bandwidth_sq.shape)
            )
    elif bandwidth_sq.ndim == 2:
        if bandwidth_sq.shape[0] == W.shape[0] and bandwidth_sq.shape[1] == W.shape[1]:  # bandwidth shape is (n_patterns, n_features)
            bandw_inv = 1 / (2 * bandwidth_sq) # (n_patterns, n_features)
            x_norm_sq = torch.matmul(torch.square(X), bandw_inv.T)  # (batch_size, n_patterns)
            w_norm_sq = torch.sum(torch.square(W) * bandw_inv, dim=1) # (n_patterns,)
            l2_norm_sq = x_norm_sq + w_norm_sq - 2.0 * torch.matmul(X, (W * bandw_inv).T)  # (batch_size, n_patterns)
            return torch.exp(-(l2_norm_sq))
        elif bandwidth_sq.shape[0] == W.shape[0] and bandwidth_sq.shape[1] == 1:  # bandwidth shape is (n_patterns, 1) per-class
            bandw_inv = 1 / (2 * bandwidth_sq.squeeze()) # (n_patterns,)
            x_norm_sq = torch.square(X).sum(dim=1, keepdim=True)  # (batch_size, 1)
            w_norm_sq = torch.square(W).sum(dim=1) # (n_patterns,)
            l2_norm_sq = x_norm_sq + w_norm_sq - 2.0 * torch.matmul(X, W.T)  # (batch_size, n_patterns)
            return torch.exp(-(l2_norm_sq * bandw_inv))
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
