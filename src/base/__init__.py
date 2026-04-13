"""Base utilities, kernels, and optimization helpers."""

from base.kernels import KERNEL_REGISTRY, KERNEL_T_REGISTRY, gaussian_kernel, gaussian_kernel_t
from base.optim import BandwidthOptimizer, log_likelihood_ratio_loss
from base.types import KernelCallable
from base.utils import normalize_l2

__all__ = [
    "BandwidthOptimizer",
    "KERNEL_REGISTRY",
    "KERNEL_T_REGISTRY",
    "KernelCallable",
    "gaussian_kernel",
    "gaussian_kernel_t",
    "log_likelihood_ratio_loss",
    "normalize_l2",
]
