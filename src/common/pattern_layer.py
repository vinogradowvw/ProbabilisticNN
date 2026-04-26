from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import nn
import torch
import numpy as np

from base.utils import normalize_l2
from base.kernels import __resolve_kernel as resolve_kernel


class PatternLayer(TransformerMixin, BaseEstimator):
    """Map input samples to kernel responses over the reference set.

    Класс преобразования входных данных в матрицу ядра с использованием

    - bandwidth: float or array-like with shape(batch_size, n_patterns, n_features)
    - kernel: str
    - normalize: bool
    """

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
    """Map input samples to kernel responses over the reference set.

    Класс преобразования входных данных в матрицу ядра с использованием оптимизации по значению ширины

    Uses optimization over a loss function for the bandwidth parameters
    Possible parameters sharing types:
    - per class bandwidth
    - per feature bandwidth
    - per class per feature bandwidth

    Возможные типы параметризации параметров ширины:
    - отдельная ширина для каждого класса
    - отдельная ширина для каждого признака
    - отдельная ширина для каждого класса по каждому признаку
    """

    def __init__(
        self,
        kernel,
        bandwidth_sharing="per_feature",
        normalize=True
    ) -> None:
        self.bandwidth_sharing = bandwidth_sharing
        self.kernel = kernel
        self.normalize = normalize

    def fit(self, X, y=None):
        X = validate_data(self, X)

        self.kernel_ = resolve_kernel(self.kernel)
        self.kernel_t_ = resolve_kernel(self.kernel, torch=True)

        if self.normalize:
            self.patterns_ = normalize_l2(X)
        else:
            self.patterns_ = X

        self.patterns_t_ = torch.as_tensor(self.patterns_, dtype=torch.float32)
        self.feature_size = X.shape[1]

        # bandwidth parameter initialization based on the bandwidth_sharing strategy
        if self.bandwidth_sharing == "per_feature":
            self.bandwidth_params = nn.Parameter(torch.ones(self.feature_size, dtype=torch.float32))
        elif self.bandwidth_sharing == "per_class":
            if y is None:
                raise ValueError("`y` is required when bandwidth_sharing='per_class'.")
            y = np.asarray(y)
            if y.ndim != 1:
                y = y.ravel()
            if y.shape[0] != X.shape[0]:
                raise ValueError("`y` must have the same number of samples as `X`.")
            self.classes_, self.y_encoded_ = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
            self.bandwidth_params = nn.Parameter(torch.ones(self.n_classes_, dtype=torch.float32))
        elif self.bandwidth_sharing == "per_class_per_feature":
            if y is None:
                raise ValueError("`y` is required when bandwidth_sharing='per_class_per_feature'.")
            y = np.asarray(y)
            if y.ndim != 1:
                y = y.ravel()
            if y.shape[0] != X.shape[0]:
                raise ValueError("`y` must have the same number of samples as `X`.")
            self.classes_, self.y_encoded_  = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
            self.bandwidth_params = nn.Parameter(torch.ones(self.n_classes_, self.feature_size, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown bandwidth_sharing={self.bandwidth_sharing}")

        return self

    def __broadcast_bandwidth(self, bandwidth_vector):
        """
        Returns bandwidth in the minimal shape required by the kernel.
        Возвращает вектор параметров ширины в минимальной форме,
        необходимой для kernel-функции:
            - (n_patterns, n_features) при bandwidth_sharing="per_class_per_feature"
            - (n_features,) при bandwidth_sharing="per_feature"
            - (n_patterns, 1) при bandwidth_sharing="per_class"

        """
        is_torch = torch.is_tensor(bandwidth_vector)

        if self.bandwidth_sharing == "per_feature":
            return bandwidth_vector
        elif self.bandwidth_sharing == "per_class":
            # per class bandwidth_vector shape (n_classes,)
            # параметр ширины для каждого класса с размером (n_classes,)
            if is_torch:
                y_idx = torch.as_tensor(self.y_encoded_, dtype=torch.long, device=bandwidth_vector.device)
                return bandwidth_vector[y_idx].reshape(-1, 1)  # (n_patterns, 1) classes aligned
            else:
                return bandwidth_vector[self.y_encoded_].reshape(-1, 1)
        elif self.bandwidth_sharing == "per_class_per_feature":
            # per class per feature bandwidth_vector shape (n_classes, n_features)
            # параметр ширины для каждого класса по каждому признаку с размером (n_classes, n_features)
            if is_torch:
                y_idx = torch.as_tensor(self.y_encoded_, dtype=torch.long, device=bandwidth_vector.device)
                return bandwidth_vector[y_idx]  # (n_patterns, n_features)  classes aligned
            else:
                return bandwidth_vector[self.y_encoded_]  # (n_patterns, n_features)  classes aligned
        else:
            raise ValueError(f"Unknown bandwidth_sharing={self.bandwidth_sharing}")

    def _loo(self):
        """Returns the kernel matrix for the Leave-One-Out (LOO)
        Возвращает матрицу значений ядра для объектов из обучающей выборки с помощью Leave-One-Out (LOO)
        """
        bandwidth = self.__broadcast_bandwidth(self.bandwidth_params)
        K = self.kernel_t_(
            self.patterns_t_,
            self.patterns_t_,
            bandwidth=bandwidth,
            normalized=self.normalize,
        )
        diagonal_mask = 1.0 - torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
        return K * diagonal_mask

    def transform(self, X):
        check_is_fitted(self, ["kernel_", "patterns_", "converged_", "bandwidth_"])
        X = validate_data(self, X, reset=False)
        if self.normalize:
            X_transformed = normalize_l2(X)
        else:
            X_transformed = X
        bandwidth = self.__broadcast_bandwidth(np.asarray(self.bandwidth_))
        return self.kernel_(
            X_transformed,
            self.patterns_,
            bandwidth=bandwidth,
            normalized=self.normalize,
        )
