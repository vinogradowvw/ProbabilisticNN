from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from common.pattern_layer import PatternLayer, AdaptivePatternLayer
from pnn.layers import OutputLayer, SummationLayer
from base.optim import BandwidthOptimizer
from base.utils import normalize_l2
from base.utils import cast_to_dtype
from base.utils import validate_backend


class PNN(ClassifierMixin, BaseEstimator):
    """Classic Probabilistic Neural Network classifier.

    Классический классификатор Probabilistic Neural Network.

    Uses a fixed kernel bandwidth for all features and all patterns.
    Использует фиксированную ширину ядра для всех признаков и всех паттернов.
    """

    def __init__(
        self,
        bandwidth=0.5,
        kernel="gaussian",
        losses="uniform",
        normalize=True,
        backend="numpy",
        compute_dtype="auto",
    ):
        validate_backend(backend)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.losses = losses
        self.normalize = normalize
        self.backend = backend
        self.compute_dtype = compute_dtype

    def fit(self, X, y):
        """Store training patterns and fit all PNN layers.

        Сохраняет обучающие паттерны и обучает все слои PNN.
        """
        X, y = validate_data(self, X, y)
        X = cast_to_dtype(X, self.compute_dtype)
        self.bandwidth = cast_to_dtype(self.bandwidth, self.compute_dtype)
        self.classes_ = unique_labels(y)
        self.y_ = y
        self.pattern_layer_ = PatternLayer(
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            normalize=self.normalize,
            backend=self.backend
        ).fit(X)

        self.summation_layer_ = SummationLayer().fit(X, y)
        self.output_layer_ = OutputLayer(self.losses).fit(y)

        return self

    def predict(self, X):
        """Predict class labels for input samples.

        Предсказывает метки классов для входных объектов.
        """
        check_is_fitted(
            self,
            ["classes_", "y_", "pattern_layer_", "summation_layer_", "output_layer_"],
        )

        X = validate_data(self, X, reset=False)
        X = cast_to_dtype(X, self.compute_dtype)

        if self.backend == "numba":
                from numba_backend import pnn_jit_inference

                X_transformed = normalize_l2(X) if self.normalize else X
                y_pred_encoded = pnn_jit_inference(
                    kernel=self.kernel,
                    X=X_transformed,
                    W=self.pattern_layer_.patterns_,
                    y_encoded=self.summation_layer_.y_encoded_,
                    n_classes=self.summation_layer_.n_classes_,
                    likelihood_multiplier=self.output_layer_.likelihood_multiplier_,
                    bandwidth=self.pattern_layer_.bandwidth_,
                    bandwidth_sharing="scalar",
                    normalized=self.normalize,
                )
                return self.output_layer_.classes_[y_pred_encoded]

        K = self.pattern_layer_.transform(X)
        f = self.summation_layer_.transform(K)
        return self.output_layer_.transform(f)

    def predict_proba(self, X):
        """Predict posterior class probabilities for input samples.

        Предсказывает апостериорные вероятности классов для входных объектов.
        """
        check_is_fitted(
            self,
            ["classes_", "y_", "pattern_layer_", "summation_layer_", "output_layer_"],
        )

        X = validate_data(self, X, reset=False)
        X = cast_to_dtype(X, self.compute_dtype)

        K = self.pattern_layer_.transform(X)
        f = self.summation_layer_.transform(K)
        posteriori = self.output_layer_.posteriori(f)
        return posteriori


class AdaptivePNN(PNN):
    """Adaptive Probabilistic Neural Network

    Uses optimization over a loss function for the bandwidth parameters
    Использует оптимизацию loss-функции по параметрам ширины ядра

    Available parameters sharing types:
    - per class bandwidth
    - per feature bandwidth
    - per class per feature bandwidth

    Доступные типы совместного использования параметров:
    - ширина на класс
    - ширина на признак
    - ширина на класс и признак
    """
    def __init__(
        self,
        kernel="gaussian",
        losses="uniform",
        loss="correct_class_probability",
        bandwidth_sharing="per_feature",
        max_iter=100,
        solver="auto",
        solver_options=None,
        normalize=True,
        backend="numpy",
        compute_dtype="auto",
    ):
        super().__init__(
            bandwidth=0,
            kernel=kernel,
            losses=losses,
            normalize=normalize,
            backend=backend,
            compute_dtype=compute_dtype,
        )
        self.loss = loss
        self.bandwidth_sharing = bandwidth_sharing
        self.max_iter = max_iter
        self.eval_mode = False
        self.normalize = normalize
        self.solver = solver
        self.solver_options = solver_options
        self.backend = backend
        self.compute_dtype = compute_dtype

    def fit(
        self,
        X,
        y
    ):
        """Fit AdaptivePNN layers and optimize bandwidth parameters.

        Обучает слои AdaptivePNN и оптимизирует параметры ширины.
        """
        X, y = validate_data(self, X, y)
        X = cast_to_dtype(X, self.compute_dtype)
        self.classes_ = unique_labels(y)
        self.y_ = y

        # Adaptive pattern layer owns trainable bandwidth parameters.
        # Adaptive pattern layer хранит обучаемые параметры ширины.
        self.pattern_layer_ = AdaptivePatternLayer(
            kernel=self.kernel,
            bandwidth_sharing=self.bandwidth_sharing,
            normalize=self.normalize,
            backend=self.backend,
            compute_dtype=self.compute_dtype,
        ).fit(X, y)
        self.summation_layer_ = SummationLayer().fit(X, y)
        self.output_layer_ = OutputLayer(self.losses).fit(y)

        # Bandwidth optimizer runs the LOO training objective.
        # Оптимизатор ширин запускает обучающую LOO-цель.
        self.optimizer_ = BandwidthOptimizer(
            model=self,
            loss=self.loss,
            max_iter=self.max_iter,
            solver=self.solver,
            solver_options=self.solver_options,
        ).optimize()
        self.bandwidth_ = self.optimizer_.bandwidth_

        return self

    def _forward_train(self, bandwidth=None, return_proba=False, return_encoded=False):
        """Run Leave-One-Out forward pass on the training set.

        Выполняет Leave-One-Out forward pass на обучающей выборке.
        """
        if return_proba and return_encoded:
            raise ValueError("`return_proba` and `return_encoded` cannot both be True.")

        K_loo = self.pattern_layer_._loo(bandwidth)
        f = self.summation_layer_.transform(K_loo)
        if return_proba:
            out = self.output_layer_.posteriori(f)
        elif return_encoded:
            out = self.output_layer_.transform_encoded(f)
        else:
            out = self.output_layer_.transform(f)

        return out

    def predict_proba(self, X):
        """Predict posterior class probabilities with optimized bandwidths.

        Предсказывает апостериорные вероятности с оптимизированными ширинами.
        """
        check_is_fitted(
            self,
            [
                "classes_",
                "y_",
                "pattern_layer_",
                "summation_layer_",
                "output_layer_",
            ],
        )

        X = validate_data(self, X, reset=False)
        X = cast_to_dtype(X, self.compute_dtype)

        K = self.pattern_layer_.transform(X)
        f = self.summation_layer_.transform(K)
        posteriori = self.output_layer_.posteriori(f)
        return posteriori

    def predict(self, X):
        """Predict class labels with optimized bandwidths.

        Предсказывает метки классов с оптимизированными ширинами.
        """
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

        X = validate_data(self, X, reset=False)
        X = cast_to_dtype(X, self.compute_dtype)

        if self.backend == "numba":
            from numba_backend import pnn_jit_inference

            X_transformed = normalize_l2(X) if self.normalize else X
            bandwidth = self.pattern_layer_._prepare_bandwidth(self.pattern_layer_.bandwidth_)
            y_pred_encoded = pnn_jit_inference(
                kernel=self.kernel,
                X=X_transformed,
                W=self.pattern_layer_.patterns_,
                y_encoded=self.summation_layer_.y_encoded_,
                n_classes=self.summation_layer_.n_classes_,
                likelihood_multiplier=self.output_layer_.likelihood_multiplier_,
                bandwidth=bandwidth,
                bandwidth_sharing=self.pattern_layer_.bandwidth_sharing,
                normalized=self.normalize,
            )
            return self.output_layer_.classes_[y_pred_encoded]

        K = self.pattern_layer_.transform(X)
        f = self.summation_layer_.transform(K)
        return self.output_layer_.transform(f)
