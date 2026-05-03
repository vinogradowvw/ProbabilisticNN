"""Тесты для PNN и AdaptivePNN estimator-ов."""
import numpy as np
import pytest

import probabilisticnn.pnn.pnn as pnn_module
from probabilisticnn.pnn.pnn import AdaptivePNN
from probabilisticnn.pnn.pnn import PNN
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator


FLOAT_DTYPES = (np.float32, np.float64)

KERNELS = {"gaussian", "laplacian", "exponential"}
LOSSES = {"correct_class_probability", "bce", "cross_entropy"}
SOLVERS = {"auto", "lbfgs", "slsqp", "nelder_mead", "powell"}
BACKENDS = {"numpy", "numba"}

def _compute_dtype_name(dtype) -> str:
    """Возвращает имя dtype в формате строкового API библиотеки."""
    return np.dtype(dtype).name


def _classification_dataset(dtype=np.float64):
    """Небольшой separable dataset для проверки классификации."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.0, 10.1],
        ],
        dtype=dtype,
    )
    y_int = np.array([0, 0, 1, 1])
    y_str = np.array(["class_a", "class_a", "class_b", "class_b"])
    return X, y_int, y_str


class DummyBandwidthOptimizer:
    """Stub оптимизатора, который позволяет тестировать AdaptivePNN изолированно."""

    instances = []

    def __init__(self, model, loss, max_iter, solver, solver_options):
        self.model = model
        self.loss = loss
        self.max_iter = max_iter
        self.solver = solver
        self.solver_options = solver_options
        self.bandwidth_ = None
        self.converged_ = True
        # sklearn expects estimators with max_iter to expose at least one executed iteration.
        self.n_iter_ = 1
        DummyBandwidthOptimizer.instances.append(self)

    def optimize(self):
        bandwidth_params = np.asarray(self.model.pattern_layer_.bandwidth_params)
        bandwidth = np.full_like(bandwidth_params, 0.2)

        self.bandwidth_ = bandwidth.copy()
        self.model.pattern_layer_.bandwidth_ = bandwidth.copy()
        self.model.pattern_layer_.bandwidth_params = bandwidth.copy()
        self.model.pattern_layer_.converged_ = True
        self.model.pattern_layer_.n_iter_ = self.n_iter_
        self.model.bandwidth_ = bandwidth.copy()
        return self


class TestPNN:
    """Проверка базового PNN estimator-а."""

    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("normalize", [True, False])
    def test_sklearn_compatibility(self, kernel, backend, normalize):
        """Проверка совместимости с scikit-learn."""
        pnn = PNN(kernel=kernel, backend=backend, normalize=normalize)
        check_estimator(pnn)

    def test_predict_returns_expected_string_labels(self):
        """predict должен возвращать исходные label-ы, а не encoded индексы классов."""
        X, _, y = _classification_dataset(dtype=np.float64)
        X_test = np.array([[0.0, 0.05], [10.0, 10.05]], dtype=np.float64)

        model = PNN(bandwidth=0.2, kernel="gaussian", normalize=False).fit(X, y)
        actual = model.predict(X_test)

        np.testing.assert_array_equal(actual, np.array(["class_a", "class_b"]))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_predict_proba_returns_normalized_probabilities_and_preserves_dtype(self, dtype):
        """predict_proba должен возвращать нормированные вероятности в compute dtype."""
        X, y, _ = _classification_dataset(dtype=dtype)
        X_test = np.array([[0.0, 0.05], [10.0, 10.05]], dtype=dtype)

        model = PNN(
            bandwidth=np.dtype(dtype).type(0.2),
            kernel="gaussian",
            normalize=False,
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X, y)
        actual = model.predict_proba(X_test)

        assert actual.shape == (2, 2)
        np.testing.assert_allclose(actual.sum(axis=1), np.ones(2, dtype=dtype), atol=1e-6)
        np.testing.assert_array_equal(np.argmax(actual, axis=1), np.array([0, 1]))
        assert actual.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("method_name", ["predict", "predict_proba"])
    def test_predict_methods_raise_before_fit(self, method_name):
        """Estimator обязан следовать sklearn-контракту и падать до fit."""
        model = PNN(bandwidth=0.2, kernel="gaussian", normalize=False)

        with pytest.raises(NotFittedError):
            getattr(model, method_name)(np.array([[0.0, 0.0]], dtype=np.float64))


class TestAdaptivePNN:
    """Проверка AdaptivePNN без зависимости от реальной SciPy-оптимизации."""

    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("loss", LOSSES)
    @pytest.mark.parametrize("solver", SOLVERS)
    @pytest.mark.parametrize("bandwidth_sharing", ["per_feature", "per_class", "per_class_per_feature"])
    def test_adaptive_pnn_sklearn_compatibility(self, kernel, backend, normalize, loss, solver, bandwidth_sharing):
        """Проверка совместимости с scikit-learn."""
        pnn = AdaptivePNN(kernel=kernel, backend=backend, normalize=normalize, loss=loss, solver=solver, bandwidth_sharing=bandwidth_sharing)
        check_estimator(pnn)

    @pytest.fixture(autouse=True)
    def _patch_optimizer(self, monkeypatch):
        """Подменяет реальный BandwidthOptimizer управляемым тестовым stub-ом."""
        DummyBandwidthOptimizer.instances.clear()
        monkeypatch.setattr(pnn_module, "BandwidthOptimizer", DummyBandwidthOptimizer)

    @pytest.mark.parametrize(
        ("sharing", "expected_shape"),
        [
            ("per_feature", (2,)),
            ("per_class", (2,)),
            ("per_class_per_feature", (2, 2)),
        ],
    )
    def test_fit_uses_optimizer_and_exposes_bandwidth(self, sharing, expected_shape):
        """fit должен прокинуть параметры в оптимизатор и сохранить найденную ширину."""
        X, y, _ = _classification_dataset(dtype=np.float64)

        model = AdaptivePNN(
            kernel="gaussian",
            bandwidth_sharing=sharing,
            loss="correct_class_probability",
            max_iter=7,
            solver="powell",
            solver_options={"xtol": 1e-3},
            normalize=False,
            compute_dtype="float64",
        ).fit(X, y)

        optimizer = DummyBandwidthOptimizer.instances[0]
        assert optimizer.model is model
        assert optimizer.loss == "correct_class_probability"
        assert optimizer.max_iter == 7
        assert optimizer.solver == "powell"
        assert optimizer.solver_options == {"xtol": 1e-3}
        assert model.bandwidth_.shape == expected_shape
        assert model.pattern_layer_.bandwidth_.shape == expected_shape
        assert model.pattern_layer_.converged_ is True
        assert model.n_iter_ == optimizer.n_iter_
        assert model.converged_ is optimizer.converged_
        assert model.optimizer_ is optimizer

    def test_forward_train_rejects_conflicting_flags(self):
        """Внутренний train-path не должен одновременно возвращать proba и encoded."""
        model = AdaptivePNN()

        with pytest.raises(ValueError, match="cannot both be True"):
            model._forward_train(return_proba=True, return_encoded=True)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward_train_returns_labels_encoded_and_proba(self, dtype):
        """_forward_train должен уметь возвращать три представления одного вывода."""
        X, _, y = _classification_dataset(dtype=dtype)

        model = AdaptivePNN(
            kernel="gaussian",
            bandwidth_sharing="per_feature",
            normalize=False,
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X, y)

        labels = model._forward_train()
        encoded = model._forward_train(return_encoded=True)
        proba = model._forward_train(return_proba=True)

        np.testing.assert_array_equal(labels, y)
        np.testing.assert_array_equal(encoded, np.array([0, 0, 1, 1]))
        assert proba.shape == (4, 2)
        np.testing.assert_array_equal(np.argmax(proba, axis=1), encoded)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(4, dtype=dtype), atol=1e-6)
        assert proba.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_predict_and_predict_proba_work_after_fit(self, dtype):
        """После fit адаптивная модель должна работать как обычный classifier API."""
        X, _, y = _classification_dataset(dtype=dtype)
        X_test = np.array([[0.0, 0.05], [10.0, 10.05]], dtype=dtype)

        model = AdaptivePNN(
            kernel="gaussian",
            bandwidth_sharing="per_feature",
            normalize=False,
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X, y)
        labels = model.predict(X_test)
        proba = model.predict_proba(X_test)

        np.testing.assert_array_equal(labels, np.array(["class_a", "class_b"]))
        assert proba.shape == (2, 2)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(2, dtype=dtype), atol=1e-6)
        np.testing.assert_array_equal(np.argmax(proba, axis=1), np.array([0, 1]))
        assert proba.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("method_name", ["predict", "predict_proba"])
    def test_predict_methods_raise_before_fit(self, method_name):
        """AdaptivePNN тоже обязан соблюдать NotFittedError-контракт sklearn."""
        model = AdaptivePNN()

        with pytest.raises(NotFittedError):
            getattr(model, method_name)(np.array([[0.0, 0.0]], dtype=np.float64))
