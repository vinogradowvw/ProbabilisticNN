from base.optim import BandwidthOptimizer
from common import AdaptivePatternLayer
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from common.pattern_layer import PatternLayer
from grnn.summation_layer import SummationLayer


class GRNN(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        bandwidth,
        kernel
    ) -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.pattern_layer_ = PatternLayer(
            self.bandwidth,
            self.kernel,
            normalize=False,
        ).fit(X)

        self.summation_layer_ = SummationLayer().fit(y)

        return self
    
    def predict(self, X):
        check_is_fitted(self, ["pattern_layer_", "summation_layer_"])
        X = validate_data(self, X, reset=False)
        K = self.pattern_layer_.transform(X)
        out = self.summation_layer_.transform(K)
        return out


class AdaptiveGRNN(GRNN):

    def __init__(
        self, 
        kernel="gaussian",
        loss="mse",
        lr=1e-2,
        eps=1e-12,
        max_iter=100,
        tol=1e-4,
        min_bandwidth=1e-6,
        normalize=False,
        verbose=False,
    ) -> None:
        self.lr = lr
        self.loss = loss
        self.eps = eps
        self.kernel = kernel
        self.max_iter = max_iter
        self.tol = tol
        self.min_bandwidth = min_bandwidth
        self.normalize = normalize
        self.verbose = verbose
        self.bandwidth_sharing = "per_feature"

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.pattern_layer_ = AdaptivePatternLayer(
            kernel=self.kernel,
            bandwidth_sharing=self.bandwidth_sharing,
            normalize=self.normalize,
        ).fit(X)
        self.summation_layer_ = SummationLayer().fit(y)
        self.optimizer_ = BandwidthOptimizer(
            model=self,
            loss=self.loss,
            lr=self.lr,
            max_iter=self.max_iter,
            tol=self.tol,
            min_bandwidth=self.min_bandwidth,
            eps=self.eps,
            verbose=self.verbose
        ).optimize()
        self.bandwidth_ = self.optimizer_.bandwidth_
        return self
    
    def predict(self, X):
        check_is_fitted(self, ["bandwidth_", "pattern_layer_", "summation_layer_", "optimizer_"])
        X = validate_data(self, X, reset=False)
        K = self.pattern_layer_.transform(X)
        out = self.summation_layer_.transform(K)
        return out

    def _forward_train(self):
        K_loo = self.pattern_layer_._loo()
        out = self.summation_layer_.transform(K_loo)
        return out
