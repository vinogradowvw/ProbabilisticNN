from probabilisticnn.base.optim import BandwidthOptimizer
from probabilisticnn.common import AdaptivePatternLayer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from probabilisticnn.common.pattern_layer import PatternLayer
from probabilisticnn.grnn.layers import SummationLayer
from probabilisticnn.base.utils import normalize_l2
from probabilisticnn.base.utils import cast_to_dtype
from probabilisticnn.base.utils import validate_backend


class GRNN(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        bandwidth=1.0,
        kernel="gaussian",
        backend="numpy",
        compute_dtype="auto"
    ) -> None:
        validate_backend(backend)

        self.bandwidth = bandwidth
        self.kernel = kernel
        self.backend = backend
        self.compute_dtype = compute_dtype

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        X = cast_to_dtype(X, self.compute_dtype)
        y = cast_to_dtype(y, self.compute_dtype)
        self.bandwidth = cast_to_dtype(self.bandwidth, self.compute_dtype)
        
        self.pattern_layer_ = PatternLayer(
            self.bandwidth,
            self.kernel,
            normalize=False,
            backend=self.backend,
        ).fit(X)

        self.summation_layer_ = SummationLayer().fit(y)

        return self
    
    def predict(self, X):
        check_is_fitted(self, ["pattern_layer_", "summation_layer_"])
        X = validate_data(self, X, reset=False)
        X = cast_to_dtype(X, self.compute_dtype)

        if self.backend == "numba":
            from probabilisticnn.numba_backend import grnn_jit_inference

            X_transformed = normalize_l2(X) if self.pattern_layer_.normalize else X
            out = grnn_jit_inference(
                kernel=self.kernel,
                X=X_transformed,
                W=self.pattern_layer_.patterns_,
                y=self.summation_layer_.y_,
                bandwidth=self.pattern_layer_.bandwidth_,
                bandwidth_sharing="scalar",
                normalized=self.pattern_layer_.normalize,
            )
            return out

        K = self.pattern_layer_.transform(X)
        out = self.summation_layer_.transform(K)
        return out


class AdaptiveGRNN(GRNN):

    def __init__(
        self, 
        kernel="gaussian",
        loss="mse",
        max_iter=100,
        solver="auto",
        solver_options=None,
        normalize=False,
        backend="numpy",
        compute_dtype="auto",
    ) -> None:
        validate_backend(backend)
        self.loss = loss
        self.kernel = kernel
        self.max_iter = max_iter
        self.normalize = normalize
        self.bandwidth_sharing = "per_feature"
        self.solver = solver
        self.solver_options = solver_options
        self.backend = backend
        self.compute_dtype = compute_dtype

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        X = cast_to_dtype(X, self.compute_dtype)
        y = cast_to_dtype(y, self.compute_dtype)

        self.pattern_layer_ = AdaptivePatternLayer(
            kernel=self.kernel,
            bandwidth_sharing=self.bandwidth_sharing,
            normalize=self.normalize,
            backend=self.backend,
            compute_dtype=self.compute_dtype,
        ).fit(X)
        self.summation_layer_ = SummationLayer().fit(y)
        self.optimizer_ = BandwidthOptimizer(
            model=self,
            loss=self.loss,
            max_iter=self.max_iter,
            solver=self.solver,
            solver_options=self.solver_options,
        ).optimize()
        self.bandwidth_ = self.optimizer_.bandwidth_
        return self
    
    def predict(self, X):
        check_is_fitted(self, ["bandwidth_", "pattern_layer_", "summation_layer_", "optimizer_"])
        X = validate_data(self, X, reset=False)
        X = cast_to_dtype(X, self.compute_dtype)

        if self.backend == "numba":
            from probabilisticnn.numba_backend import grnn_jit_inference
            X_transformed = normalize_l2(X) if self.pattern_layer_.normalize else X
            bandwidth = self.pattern_layer_._prepare_bandwidth(self.pattern_layer_.bandwidth_)
            out = grnn_jit_inference(
                kernel=self.kernel,
                X=X_transformed,
                W=self.pattern_layer_.patterns_,
                y=self.summation_layer_.y_,
                bandwidth=bandwidth,
                bandwidth_sharing=self.pattern_layer_.bandwidth_sharing,
                normalized=self.normalize,
            )
            return out

        K = self.pattern_layer_.transform(X)
        out = self.summation_layer_.transform(K)
        return out

    def _forward_train(self, bandwidth = None):
        K_loo = self.pattern_layer_._loo(bandwidth)
        out = self.summation_layer_.transform(K_loo)
        return out
