from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
import numpy as np
from probabilisticnn.base.utils import normalize_l2
from probabilisticnn.base.utils import cast_to_dtype
from probabilisticnn.base.utils import as_bandwidth_array


class PatternLayer(TransformerMixin, BaseEstimator):
    """Map input samples to kernel responses over the reference set."""

    def __init__(
        self,
        bandwidth: float = 0.5,
        kernel="gaussian",
        normalize: bool = True,
        backend="numpy"
    ) -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.normalize = normalize
        self.backend = backend

    def fit(self, X, y=None):
        from probabilisticnn.base.kernels import resolve_kernel
        self.kernel_ = resolve_kernel(self.kernel)

        self.bandwidth_ = as_bandwidth_array(self.bandwidth)
        self.patterns_ = normalize_l2(X) if self.normalize else X
        return self

    def transform(self, X):
        check_is_fitted(self, ["kernel_", "patterns_", "bandwidth_"])
        X = validate_data(self, X, reset=False)
        X_transformed = normalize_l2(X) if self.normalize else X.copy()
        return self.kernel_(
            X_transformed,
            self.patterns_,
            bandwidth=self.bandwidth_,
            bandwidth_sharing="scalar",
            normalized=self.normalize,
        )


class AdaptivePatternLayer(TransformerMixin, BaseEstimator):
    """Map input samples to kernel responses with optimizable bandwidth parameters.

    Bandwidth can be shared per feature, per class, or per class per feature.
    """

    def __init__(
        self,
        kernel,
        bandwidth_sharing="per_feature",
        normalize=True,
        backend="numpy",
        compute_dtype="auto",
    ) -> None:
        self.bandwidth_sharing = bandwidth_sharing
        self.kernel = kernel
        self.normalize = normalize
        self.backend = backend
        self.compute_dtype = compute_dtype

    def fit(self, X, y=None):
        X = validate_data(self, X)

        from probabilisticnn.base.kernels import resolve_kernel
        self.kernel_ = resolve_kernel(self.kernel)

        if self.normalize:
            self.patterns_ = normalize_l2(X)
        else:
            self.patterns_ = X

        self.feature_size = X.shape[1]

        # initialize bandwidth from data std to give optimizer a sensible starting point
        if self.bandwidth_sharing == "per_feature":
            std = cast_to_dtype(np.std(self.patterns_, axis=0), self.compute_dtype) + 1e-12
            self.bandwidth_params = std
        elif self.bandwidth_sharing == "per_class":
            if y is None:
                raise ValueError("`y` is required when bandwidth_sharing='per_class'.")
            y = np.asarray(y)
            if y.ndim != 1:
                y = y.ravel()
            if y.shape[0] != self.patterns_.shape[0]:
                raise ValueError("`y` must have the same number of samples as `X`.")
            self.classes_, self.y_encoded_ = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)

            self.bandwidth_params = cast_to_dtype(np.zeros(self.n_classes_), self.compute_dtype)
            for cl in np.unique(self.y_encoded_):
                self.bandwidth_params[cl] = (
                    cast_to_dtype(
                        np.std(
                            self.patterns_[self.y_encoded_ == cl],
                            axis=0,
                        ).mean(),
                        self.compute_dtype,
                    )
                    + 1e-12
                )
        elif self.bandwidth_sharing == "per_class_per_feature":
            if y is None:
                raise ValueError("`y` is required when bandwidth_sharing='per_class_per_feature'.")
            y = np.asarray(y)
            if y.ndim != 1:
                y = y.ravel()
            if y.shape[0] != self.patterns_.shape[0]:
                raise ValueError("`y` must have the same number of samples as `X`.")
            self.classes_, self.y_encoded_  = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)

            self.bandwidth_params = cast_to_dtype(
                np.zeros((self.n_classes_, self.feature_size)),
                self.compute_dtype,
            )
            for cl in np.unique(self.y_encoded_):
                self.bandwidth_params[cl] = (
                    cast_to_dtype(
                        np.std(
                            self.patterns_[self.y_encoded_ == cl],
                            axis=0,
                        ),
                        self.compute_dtype,
                    ) + 1e-12
                )
        else:
            raise ValueError(f"Unknown bandwidth_sharing={self.bandwidth_sharing}")
        return self

    def __broadcast_bandwidth(self, bandwidth_vector):
        """Expand class-level bandwidth to pattern-level by indexing with y_encoded_."""
        if self.bandwidth_sharing == "per_feature":
            return bandwidth_vector
        elif self.bandwidth_sharing == "per_class":
            return bandwidth_vector[self.y_encoded_]  # (n_classes,) → (n_patterns,)
        elif self.bandwidth_sharing == "per_class_per_feature":
            return bandwidth_vector[self.y_encoded_]  # (n_classes, n_features) → (n_patterns, n_features)
        else:
            raise ValueError(f"Unknown bandwidth_sharing={self.bandwidth_sharing}")

    def _prepare_bandwidth(self, bandwidth):
        """Normalize raw bandwidth parameters to the kernel-ready shape.

        Приводит параметры ширины к форме, которую ожидает kernel-функция.
        """
        bandwidth = cast_to_dtype(np.asarray(bandwidth), self.compute_dtype)
        n_patterns = self.patterns_.shape[0]

        if self.bandwidth_sharing == "per_feature":
            expected_shape = (self.feature_size,)
            if bandwidth.shape != expected_shape:
                raise ValueError(
                    f"Invalid bandwidth shape for 'per_feature': expected {expected_shape}, got {bandwidth.shape}."
                )
            return bandwidth

        if self.bandwidth_sharing == "per_class":
            raw_shape = (self.n_classes_,)
            broadcast_shape = (n_patterns, 1)
            if bandwidth.shape == raw_shape:
                return self.__broadcast_bandwidth(bandwidth)
            if bandwidth.shape == broadcast_shape:
                return bandwidth
            raise ValueError(
                f"Invalid bandwidth shape for 'per_class': expected {raw_shape} or {broadcast_shape}, got {bandwidth.shape}."
            )

        if self.bandwidth_sharing == "per_class_per_feature":
            raw_shape = (self.n_classes_, self.feature_size)
            broadcast_shape = (n_patterns, self.feature_size)
            if bandwidth.shape == raw_shape:
                return self.__broadcast_bandwidth(bandwidth)
            if bandwidth.shape == broadcast_shape:
                return bandwidth
            raise ValueError(
                "Invalid bandwidth shape for 'per_class_per_feature': "
                f"expected {raw_shape} or {broadcast_shape}, got {bandwidth.shape}."
            )

        raise ValueError(f"Unknown bandwidth_sharing={self.bandwidth_sharing}")

    def _loo(self, bandwidth=None):
        """Return the kernel matrix for Leave-One-Out: K computed on training patterns with diagonal zeroed."""
        if bandwidth is None:
            bandwidth = self.bandwidth_params
        bandwidth = self._prepare_bandwidth(bandwidth)
        K = self.kernel_(
            self.patterns_,
            self.patterns_,
            bandwidth=bandwidth,
            bandwidth_sharing=self.bandwidth_sharing,
            normalized=self.normalize,
        )
        np.fill_diagonal(K, 0.0)
        return K

    def transform(self, X):
        check_is_fitted(self, ["kernel_", "patterns_", "converged_", "bandwidth_"])
        X = validate_data(self, X, reset=False)
        if self.normalize:
            X_transformed = normalize_l2(X)
        else:
            X_transformed = X
        bandwidth = self._prepare_bandwidth(self.bandwidth_)
        return self.kernel_(
            X_transformed,
            self.patterns_,
            bandwidth=bandwidth,
            bandwidth_sharing=self.bandwidth_sharing,
            normalized=self.normalize,
        )
