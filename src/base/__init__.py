"""Base utilities, kernels, and optimization helpers."""

__all__ = [
    "BandwidthOptimizer",
    "KERNEL_REGISTRY",
    "KernelCallable",
    "gaussian_kernel",
    "normalize_l2",
]


def __getattr__(name):
    if name in {"KERNEL_REGISTRY", "gaussian_kernel"}:
        from base.kernels import KERNEL_REGISTRY, gaussian_kernel

        exports = {
            "KERNEL_REGISTRY": KERNEL_REGISTRY,
            "gaussian_kernel": gaussian_kernel,
        }
        return exports[name]

    if name == "BandwidthOptimizer":
        from base.optim import BandwidthOptimizer

        return BandwidthOptimizer

    if name == "KernelCallable":
        from base.types import KernelCallable

        return KernelCallable

    if name == "normalize_l2":
        from base.utils import normalize_l2

        return normalize_l2

    raise AttributeError(f"module 'base' has no attribute {name!r}")
