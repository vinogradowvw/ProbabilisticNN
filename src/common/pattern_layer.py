from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import nn
import torch
import numpy as np

from base.utils import normalize_l2
from base.kernels import __resolve_kernel as resolve_kernel


class PatternLayer(TransformerMixin, BaseEstimator):
    """Map input samples to kernel responses over the reference set."""

    def __init__(
        self,
        bandwidth: float = 0.5,
        kernel="gaussian",
        normalize: bool = True,
    ) -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.normalize = normalize

    def fit(self, X, y=None):
        X = validate_data(self, X)
        self.kernel_ = resolve_kernel(self.kernel)
        self.patterns_ = normalize_l2(X) if self.normalize else X
        return self

    def transform(self, X):
        check_is_fitted(self, ["kernel_", "patterns_"])
        X = validate_data(self, X, reset=False)
        X_transformed = normalize_l2(X) if self.normalize else X
        return self.kernel_(
            X_transformed,
            self.patterns_,
            bandwidth=self.bandwidth,
            normalized=self.normalize,
        )


class AdaptivePatternLayer(TransformerMixin, BaseEstimator):
    
    def __init__(
        self,
        kernel,
    ) -> None:
        self.kernel = kernel

    def fit(self, X, y=None):
        X = validate_data(self, X)
        self.kernel_ = resolve_kernel(self.kernel)
        self.kernel_t_ = resolve_kernel(self.kernel, torch=True)
        self.patterns_ = normalize_l2(X)
        self.patterns_t_ = torch.as_tensor(self.patterns_, dtype=torch.float32)
        self.feature_size = X.shape[1]
        self.bandwidth_params = nn.Parameter(torch.ones(self.feature_size, dtype=torch.float32))
        return self

    def _loo(self):
        K = self.kernel_t_(
            self.patterns_t_,
            self.patterns_t_,
            bandwidth=self.bandwidth_params,
            normalized=True,
        )
        diagonal_mask = 1.0 - torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
        return K * diagonal_mask
    
    def transform(self, X):
        check_is_fitted(self, ["kernel_", "patterns_", "converged_"])
        X = validate_data(self, X, reset=False)
        X_normalized = normalize_l2(X)
        return self.kernel_(
            X_normalized,
            self.patterns_,
            bandwidth=np.asarray(self.bandwidth_),
            normalized=True,
        )
