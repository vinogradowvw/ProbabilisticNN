"""Тесты для GRNN моделей.
"""
import numpy as np
import pytest

import probabilisticnn.grnn.grnn as grnn_module
from probabilisticnn.base.kernels import gaussian_kernel
from probabilisticnn.grnn.grnn import AdaptiveGRNN
from probabilisticnn.grnn.grnn import GRNN
from sklearn.exceptions import NotFittedError


FLOAT_DTYPES = (np.float32, np.float64)


def _compute_dtype_name(dtype) -> str:
    return np.dtype(dtype).name


def _manual_grnn_prediction(X, W, y, bandwidth):
    K = gaussian_kernel(
        np.asarray(X),
        np.asarray(W),
        bandwidth=np.asarray(bandwidth).item() if np.asarray(bandwidth).ndim == 0 else bandwidth,
        bandwidth_sharing="scalar",
        normalized=False,
    )
    denom = np.sum(K, axis=1)
    nom = np.dot(K, y)
    return np.divide(nom, denom, out=np.zeros_like(nom, dtype=K.dtype), where=denom > 0)


def _regression_dataset(dtype=np.float64):
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [10.0, 10.0],
            [10.0, 10.2],
        ],
        dtype=dtype,
    )
    y = np.array([1.0, 1.0, 3.0, 3.0], dtype=dtype)
    return X, y


def _duplicate_regression_dataset(dtype=np.float64):
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=dtype,
    )
    y = np.array([1.0, 1.0, 3.0, 3.0], dtype=dtype)
    return X, y


class DummyBandwidthOptimizer:
    """Подмена оптимизатора ширины для тестирования для правильного unit test."""
    instances = []

    def __init__(self, model, loss, max_iter, solver, solver_options):
        self.model = model
        self.loss = loss
        self.max_iter = max_iter
        self.solver = solver
        self.solver_options = solver_options
        self.bandwidth_ = None
        self.converged_ = True
        self.n_iter_ = 0
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


class TestGRNN:
    """
    Тесты для GRNN моделей.
    """
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_predict_matches_manual_kernel_weighted_average(self, dtype):
        """Проверка предсказания для GRNN

        _manual_grnn_prediction вычисляет предсказание для одного объекта с помощью ручной функции ядра. 
        (подразумевается, что функция ядра вычисляется правильно

        так как тестируется именно сам estimator (GRNN) мы не тестируем функцию ядра и проверяем только сам контракт API.
        """
        X_train = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=dtype)
        y_train = np.array([1.0, 3.0], dtype=dtype)
        X_test = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=dtype)
        bandwidth = np.dtype(dtype).type(1.0)

        model = GRNN(
            bandwidth=bandwidth,
            kernel="gaussian",
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X_train, y_train)
        actual = model.predict(X_test)
        expected = _manual_grnn_prediction(X_test, X_train, y_train, bandwidth)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)  # чисельно не ошиблись
        assert actual.dtype == np.dtype(dtype)  # вернули правильный dtype

    def test_predict_raises_before_fit(self):
        """Проверка правильности вызова ошибки NotFittedError при вызове predict до того, как модель была обучена.
        """
        model = GRNN()

        with pytest.raises(NotFittedError):
            model.predict(np.array([[0.0, 0.0]], dtype=np.float64))

    def test_invalid_backend_raises_value_error(self):
        """Проверка правильности вызова ошибки ValueError при указании неподдерживаемого backend.
        """
        with pytest.raises(ValueError, match="Unknown backend"):
            GRNN(backend="unsupported")


class TestAdaptiveGRNN:
    """Тесты для AdaptiveGRNN моделей.
    """
    @pytest.fixture(autouse=True)
    def _patch_optimizer(self, monkeypatch):
        """Подмена оптимизатора ширины"""
        DummyBandwidthOptimizer.instances.clear()
        monkeypatch.setattr(grnn_module, "BandwidthOptimizer", DummyBandwidthOptimizer)

    def test_fit_uses_optimizer_and_exposes_bandwidth(self):
        """Проверка что:
        - fit вызывает оптимизатор и передает параметры оптимизации
        - оптимизатор возвращает оптимальную ширину
        """
        X, y = _regression_dataset(dtype=np.float64)

        model = AdaptiveGRNN(
            kernel="gaussian",
            loss="mae",
            max_iter=9,
            solver="powell",
            solver_options={"xtol": 1e-3},
            normalize=False,
            compute_dtype="float64",
        ).fit(X, y)

        optimizer = DummyBandwidthOptimizer.instances[0]
        assert optimizer.model is model  # модель переданная в оптимизатор является одни объектом
        assert optimizer.loss == "mae"  # оптимизатор использует правильный loss
        assert optimizer.max_iter == 9  # оптимизатор использует правильный max_iter
        assert optimizer.solver == "powell"  # оптимизатор использует правильный solver
        assert optimizer.solver_options == {"xtol": 1e-3}  # оптимизатор использует правильные solver_options
        assert model.bandwidth_sharing == "per_feature"  # оптимизатор использует правильную bandwidth_sharing
        assert model.bandwidth_.shape == (2,)  # оптимизатор использует правильную bandwidth_ форму
        assert model.pattern_layer_.bandwidth_.shape == (2,)  # оптимизатор передал в pattern_layer_ правильную bandwidth_ форму
        assert model.pattern_layer_.converged_ is True 
        assert model.optimizer_ is optimizer  # оптимизатор передал в объект модели оптимизатора

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward_train_returns_expected_leave_one_out_regression(self, dtype):
        """Проверяем dtype консистентность в _forward_train для AdaptiveGRNN моделей."""
        X, y = _duplicate_regression_dataset(dtype=dtype)

        model = AdaptiveGRNN(
            kernel="gaussian",
            normalize=False,
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X, y)
        actual = model._forward_train()

        np.testing.assert_allclose(actual, y, rtol=1e-6, atol=1e-6)
        assert actual.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_predict_returns_expected_regression_and_preserves_dtype(self, dtype):
        """Проверяем правильность вычисления предсказания и
        dtype консистентность в predict для AdaptiveGRNN моделей.
        """
        X, y = _regression_dataset(dtype=dtype)
        X_test = np.array([[0.0, 0.1], [10.0, 10.1]], dtype=dtype)

        model = AdaptiveGRNN(
            kernel="gaussian",
            normalize=False,
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X, y)
        actual = model.predict(X_test)

        np.testing.assert_allclose(actual, np.array([1.0, 3.0], dtype=dtype), rtol=1e-5, atol=1e-5)
        assert actual.dtype == np.dtype(dtype)

    def test_predict_raises_before_fit(self):
        """Проверка правильности вызова ошибки NotFittedError при вызове predict до того, как модель была обучена.
        """
        model = AdaptiveGRNN()

        with pytest.raises(NotFittedError):
            model.predict(np.array([[0.0, 0.0]], dtype=np.float64))
