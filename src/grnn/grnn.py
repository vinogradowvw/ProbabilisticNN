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
