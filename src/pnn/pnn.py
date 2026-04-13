from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from common.pattern_layer import PatternLayer, AdaptivePatternLayer
from pnn.layers import OutputLayer, SummationLayer
from base.optim import BandwidthOptimizer


class PNN(ClassifierMixin, BaseEstimator):
    """Classic Probabilistic neural network
    """
    
    def __init__(
        self,
        bandwidth=0.5, 
        kernel="gaussian",
        losses="uniform"
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.losses = losses
    
    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        self.y_ = y

        self.pattern_layer_ = PatternLayer(
            bandwidth=self.bandwidth,
            kernel=self.kernel,
        ).fit(X)

        self.summation_layer_ = SummationLayer().fit(X, y)
        self.output_layer_ = OutputLayer(self.losses).fit(y)

        return self
    
    def predict(self, X):
        check_is_fitted(
            self,
            ["classes_", "y_", "pattern_layer_", "summation_layer_", "output_layer_"],
        )

        K = self.pattern_layer_.transform(X)
        f = self.summation_layer_.transform(K)
        return self.output_layer_.transform(f)
    
    def predict_proba(self, X):
        check_is_fitted(
            self,
            ["classes_", "y_", "pattern_layer_", "summation_layer_", "output_layer_"],
        )
        K = self.pattern_layer_.transform(X)
        f = self.summation_layer_.transform(K)
        posteriori = self.output_layer_.posteriori(f)
        return posteriori


class AdaptivePNN(PNN):
    """Adaptive Probabilistic Neural Network

    Uses optimization over a loss function for the bandwidth parameters
    """
    def __init__(
        self,
        kernel="gaussian",
        losses="uniform",
        loss="log_likelihood_ratio",
        lr=1e-2,
        max_iter=100,
        tol=1e-4,
        min_bandwidth=1e-6,
        eps=1e-12,
        verbose=False,
    ):
        super().__init__(bandwidth=0, kernel=kernel, losses=losses)
        self.loss = loss
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.min_bandwidth = min_bandwidth
        self.eps = eps
        self.eval_mode = False
        self.verbose = verbose

    def fit(
        self,
        X,
        y
    ):
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        self.y_ = y

        self.pattern_layer_ = AdaptivePatternLayer(
            kernel=self.kernel,
        ).fit(X)
        self.summation_layer_ = SummationLayer().fit(X, y)
        self.output_layer_ = OutputLayer(self.losses).fit(y)


        self.optimizer_ = BandwidthOptimizer(
            self,
            self.loss,
            self.lr,
            self.max_iter,
            self.tol,
            self.min_bandwidth,
            self.eps,
            self.verbose
        )

        self.optimizer_.optimize()
        
        return self

    def _forvard_train(self, return_proba=False):
        K_loo = self.pattern_layer_._loo()
        f = self.summation_layer_.transform(K_loo)
        if return_proba:
            out = self.output_layer_.posteriori(f)
        else:
            out = self.output_layer_.transform(f)

        return out

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "classes_",
                "y_",
                "pattern_layer_",
                "summation_layer_",
                "output_layer_",
                "optimizer_",
            ],
        )

        K = self.pattern_layer_.transform(X)
        f = self.summation_layer_.transform(K)
        return self.output_layer_.transform(f)
