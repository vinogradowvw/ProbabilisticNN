"""Optional numba-backed inference for ProbabilisticNN."""

from numba_backend.inference import pnn_jit_inference
from numba_backend.inference import grnn_jit_inference

__all__ = [
    "pnn_jit_inference",
    "grnn_jit_inference",
]
