"""Base utilities, kernels, and optimization helpers."""

from base.kernels import KERNEL_REGISTRY, gaussian_kernel
from base.optim import BandwidthOptimizer
from base.types import KernelCallable
from base.utils import normalize_l2

__all__ = [
    "BandwidthOptimizer",
    "KERNEL_REGISTRY",
    "KernelCallable",
    "gaussian_kernel",
    "normalize_l2",
]
