"""Тесты для BandwidthOptimizer и его внутренних helper-ов."""

from types import SimpleNamespace

import numpy as np
import pytest

import probabilisticnn.base.optim as optim_module
from probabilisticnn.base.optim import BandwidthOptimizer
from probabilisticnn.base.optim import INVALID_OBJECTIVE
from probabilisticnn.base.optim import _resolve_solver


def _make_pattern_layer(bandwidth_sharing, bandwidth_params, *, pattern_dtype=np.float64):
    """Строит минимальный pattern layer stub с нужной формой bandwidth-параметров."""
    bandwidth_params = np.asarray(bandwidth_params, dtype=np.float64)

    if bandwidth_sharing == "per_feature":
        feature_size = bandwidth_params.size
        n_classes = 2
    elif bandwidth_sharing == "per_class":
        feature_size = 2
        n_classes = bandwidth_params.size
    elif bandwidth_sharing == "per_class_per_feature":
        n_classes, feature_size = bandwidth_params.shape
    else:
        feature_size = 2
        n_classes = 2

    return SimpleNamespace(
        bandwidth_sharing=bandwidth_sharing,
        bandwidth_params=bandwidth_params.copy(),
        feature_size=feature_size,
        n_classes_=n_classes,
        patterns_=np.zeros((4, feature_size), dtype=pattern_dtype),
    )


# ------------------------------------------------------------------------------
# классы для прокидывания в оптимизатор вместо настоящих эстиматоров
# ------------------------------------------------------------------------------
class DummyOutputLayer:
    """Минимальный stub выходного слоя PNN для unit-тестов оптимизатора."""

    def __init__(self, y_encoded, posterior=None):
        self.y_encoded_ = np.asarray(y_encoded, dtype=np.intp)
        self._posterior = posterior
        self.posteriori_calls = []

    def posteriori(self, f):
        f = np.asarray(f)
        self.posteriori_calls.append(f.copy())

        if self._posterior is not None:
            return np.asarray(self._posterior, dtype=f.dtype)

        denom = np.sum(f, axis=1, keepdims=True)
        return np.divide(f, denom, out=np.zeros_like(f), where=denom > 0)


class DummyPNNModel:
    """Минимальная PNN-модель, достаточная для проверки objective и optimize."""

    def __init__(
        self,
        pattern_layer,
        *,
        compute_dtype="auto",
        y_pred=None,
        last_f=None,
        posterior=None,
    ):
        self.compute_dtype = compute_dtype
        self.pattern_layer_ = pattern_layer
        self.summation_layer_ = SimpleNamespace(last_f_=None)
        self.output_layer_ = DummyOutputLayer(
            y_encoded=np.array([1, 0], dtype=np.intp),
            posterior=posterior,
        )
        self.bandwidth_ = None
        self.forward_calls = []
        self._y_pred = np.array([1, 0], dtype=np.intp) if y_pred is None else np.asarray(y_pred)
        self._last_f = (
            np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float64)
            if last_f is None
            else np.asarray(last_f, dtype=np.float64)
        )

    def _forward_train(self, bandwidth, return_proba=False, return_encoded=False):
        self.forward_calls.append(
            (
                np.asarray(bandwidth, dtype=np.float64).copy(),
                return_proba,
                return_encoded,
            )
        )
        self.summation_layer_.last_f_ = self._last_f.copy()
        return self._y_pred.copy()


class DummyGRNNModel:
    """Минимальная GRNN-модель, достаточная для проверки objective и optimize."""

    def __init__(self, pattern_layer, *, compute_dtype="auto", y_true=None, y_pred=None):
        self.compute_dtype = compute_dtype
        self.pattern_layer_ = pattern_layer
        self.summation_layer_ = SimpleNamespace(
            y_=np.array([1.0, 2.0, 4.0], dtype=np.float64)
            if y_true is None
            else np.asarray(y_true, dtype=np.float64)
        )
        self.bandwidth_ = None
        self.forward_calls = []
        self._y_pred = (
            np.array([1.5, 1.5, 3.5], dtype=np.float64)
            if y_pred is None
            else np.asarray(y_pred, dtype=np.float64)
        )

    def _forward_train(self, bandwidth):
        self.forward_calls.append(np.asarray(bandwidth, dtype=np.float64).copy())
        return self._y_pred.copy()


def _make_optimizer(
    monkeypatch,
    model,
    loss_fn,
    *,
    is_pnn_model,
    solver="auto",
    max_iter=25,
    solver_options=None,
):
    """Создает оптимизатор с подменой resolve_loss и ветки PNN/GRNN."""
    monkeypatch.setattr(optim_module, "_resolve_loss", lambda loss, model_arg: loss_fn)
    optimizer = BandwidthOptimizer(
        model=model,
        loss="dummy",
        max_iter=max_iter,
        solver=solver,
        solver_options=solver_options,
    )
    monkeypatch.setattr(optimizer, "_is_pnn_model", lambda: is_pnn_model)
    return optimizer


class TestBandwidthOptimizerHelpers:
    """Тесты для служебных функций оптимизатора."""

    @pytest.mark.parametrize(
        ("solver", "expected"),
        [
            ("auto", "L-BFGS-B"),
            ("lbfgs", "L-BFGS-B"),
            ("slsqp", "SLSQP"),
            ("nelder_mead", "Nelder-Mead"),
            ("powell", "Powell"),
        ],
    )
    def test_resolve_solver_aliases(self, solver, expected):
        """Пользовательские алиасы solver должны разворачиваться в имена SciPy-методов."""
        assert _resolve_solver(solver) == expected

    def test_resolve_solver_rejects_unknown_name(self):
        """Оптимизатор должен вызывать ошибку при указании неподдерживаемого solver."""
        with pytest.raises(ValueError, match="Unknown solver"):
            _resolve_solver("unsupported")

    @pytest.mark.parametrize(
        ("sharing", "bandwidth_params", "expected_packed", "theta", "expected_unpacked"),
        [
            (
                "per_feature",
                np.array([0.5, 1.5], dtype=np.float64),
                np.array([0.5, 1.5], dtype=np.float64),
                np.array([2.0, 3.0], dtype=np.float64),
                np.array([2.0, 3.0], dtype=np.float64),
            ),
            (
                "per_class",
                np.array([0.5, 1.5, 2.5], dtype=np.float64),
                np.array([0.5, 1.5, 2.5], dtype=np.float64),
                np.array([2.0, 3.0, 4.0], dtype=np.float64),
                np.array([2.0, 3.0, 4.0], dtype=np.float64),
            ),
            (
                "per_class_per_feature",
                np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64),
                np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64),
                np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64),
                np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64),
            ),
        ],
    )
    def test_pack_and_unpack_bandwidth_follow_sharing_strategy(
        self,
        monkeypatch,
        sharing,
        bandwidth_params,
        expected_packed,
        theta,
        expected_unpacked,
    ):
        """Проверка, что pack/unpack согласованы со схемой bandwidth_sharing."""
        pattern_layer = _make_pattern_layer(sharing, bandwidth_params)
        model = DummyGRNNModel(pattern_layer)
        optimizer = _make_optimizer(monkeypatch, model, lambda y_pred, y_true: 0.0, is_pnn_model=False)

        packed = optimizer._pack_bandwidth()
        unpacked = optimizer._unpack_bandwidth(theta)

        np.testing.assert_allclose(packed, expected_packed, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(unpacked, expected_unpacked, rtol=1e-10, atol=1e-12)

        packed[...] = -1.0
        assert np.all(model.pattern_layer_.bandwidth_params > 0)

class TestBandwidthOptimizerObjective:
    """Проверка целевой функции оптимизатора."""
    @pytest.mark.parametrize(
        "theta",
        [
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([-1.0, 1.0], dtype=np.float64),
            np.array([np.nan, 1.0], dtype=np.float64),
            np.array([np.inf, 1.0], dtype=np.float64),
        ],
    )
    def test_objective_returns_invalid_for_non_positive_or_non_finite_bandwidth(self, monkeypatch, theta):
        """Невалидная ширина должна отбрасываться до вызова model._forward_train."""
        pattern_layer = _make_pattern_layer("per_feature", np.array([0.5, 1.5], dtype=np.float64))
        model = DummyGRNNModel(pattern_layer)
        optimizer = _make_optimizer(monkeypatch, model, lambda y_pred, y_true: 0.0, is_pnn_model=False)

        assert optimizer._objective(theta) == INVALID_OBJECTIVE
        assert model.forward_calls == []

    def test_objective_uses_pnn_targets_predictions_scores_and_posteriors(self, monkeypatch):
        """В PNN-ветке objective должен передать в loss все классификационные артефакты."""
        pattern_layer = _make_pattern_layer("per_feature", np.array([0.5, 1.5], dtype=np.float64))
        posterior = np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float64)
        model = DummyPNNModel(pattern_layer, posterior=posterior)
        captured = {}

        def fake_loss(y_true, y_pred, f, proba):
            captured["y_true"] = np.asarray(y_true).copy()
            captured["y_pred"] = np.asarray(y_pred).copy()
            captured["f"] = np.asarray(f).copy()
            captured["proba"] = np.asarray(proba).copy()
            return 0.125

        optimizer = _make_optimizer(monkeypatch, model, fake_loss, is_pnn_model=True)
        theta = np.array([0.8, 1.6], dtype=np.float64)

        actual = optimizer._objective(theta)

        assert actual == pytest.approx(0.125)
        assert model.forward_calls[0][1:] == (False, True)
        np.testing.assert_array_equal(captured["y_true"], model.output_layer_.y_encoded_)
        np.testing.assert_array_equal(captured["y_pred"], model._y_pred)
        np.testing.assert_allclose(captured["f"], model._last_f, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(captured["proba"], posterior, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            model.output_layer_.posteriori_calls[0],
            model._last_f,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_objective_uses_grnn_predictions_and_targets(self, monkeypatch):
        """В GRNN-ветке objective должен передавать в loss только y_pred и y_true."""
        pattern_layer = _make_pattern_layer("per_feature", np.array([0.5, 1.5], dtype=np.float64))
        model = DummyGRNNModel(pattern_layer)
        captured = {}

        def fake_loss(y_pred, y_true):
            captured["y_pred"] = np.asarray(y_pred).copy()
            captured["y_true"] = np.asarray(y_true).copy()
            return 0.375

        optimizer = _make_optimizer(monkeypatch, model, fake_loss, is_pnn_model=False)
        theta = np.array([0.8, 1.6], dtype=np.float64)

        actual = optimizer._objective(theta)

        assert actual == pytest.approx(0.375)
        np.testing.assert_allclose(model.forward_calls[0], theta, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(captured["y_pred"], model._y_pred, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(captured["y_true"], model.summation_layer_.y_, rtol=1e-10, atol=1e-12)

    def test_objective_returns_invalid_for_non_finite_loss(self, monkeypatch):
        """Не числовой loss нельзя отдавать оптимизатору SciPy как валидную цель."""
        pattern_layer = _make_pattern_layer("per_feature", np.array([0.5, 1.5], dtype=np.float64))
        model = DummyGRNNModel(pattern_layer)
        optimizer = _make_optimizer(monkeypatch, model, lambda y_pred, y_true: np.nan, is_pnn_model=False)

        actual = optimizer._objective(np.array([0.8, 1.6], dtype=np.float64))

        assert actual == INVALID_OBJECTIVE


class TestBandwidthOptimizerOptimize:
    """Проверка публичного метода optimize."""

    def test_optimize_calls_minimize_and_updates_model_state(self, monkeypatch):
        """optimize должен вызвать SciPy minimize и синхронизировать состояние модели."""
        pattern_layer = _make_pattern_layer(
            "per_class_per_feature",
            np.array([[0.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            pattern_dtype=np.float32,
        )
        model = DummyGRNNModel(pattern_layer, compute_dtype="auto")
        optimizer = _make_optimizer(
            monkeypatch,
            model,
            lambda y_pred, y_true: 0.0,
            is_pnn_model=False,
            solver="slsqp",
            max_iter=17,
            solver_options={"ftol": 1e-6},
        )
        captured = {}
        result_x = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64)

        def fake_minimize(fun, theta0, method, bounds, options):
            captured["theta0"] = np.asarray(theta0).copy()
            captured["method"] = method
            captured["bounds"] = list(bounds)
            captured["options"] = dict(options)
            captured["objective_at_theta0"] = fun(theta0)
            return SimpleNamespace(
                x=result_x,
                nit=7,
                success=True,
                message="ok",
            )

        # Вместо реального SciPy minimize подставляем управляемый stub.
        monkeypatch.setattr(optim_module, "minimize", fake_minimize)

        actual = optimizer.optimize()

        assert actual is optimizer
        np.testing.assert_allclose(
            captured["theta0"],
            np.array([1e-4, 2.0, 3.0, 4.0], dtype=np.float64),
            rtol=1e-10,
            atol=1e-12,
        )
        assert captured["method"] == "SLSQP"
        assert captured["bounds"] == [(1e-4, None)] * 4
        assert captured["options"] == {"maxiter": 17, "ftol": 1e-6}
        assert captured["objective_at_theta0"] == pytest.approx(0.0)

        expected_bandwidth = np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(optimizer.bandwidth_, expected_bandwidth, rtol=1e-6, atol=1e-7)
        assert optimizer.bandwidth_.dtype == np.float32
        assert optimizer.solver_ == "SLSQP"
        assert optimizer.n_iter_ == 7
        assert optimizer.converged_ is True
        assert optimizer.optimization_result_.x is result_x

        np.testing.assert_allclose(model.pattern_layer_.bandwidth_, expected_bandwidth, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(model.pattern_layer_.bandwidth_params, expected_bandwidth, rtol=1e-6, atol=1e-7)
        assert model.pattern_layer_.bandwidth_.dtype == np.float32
        assert model.pattern_layer_.bandwidth_params.dtype == np.float32
        assert model.pattern_layer_.bandwidth_params is not optimizer.bandwidth_
        assert model.pattern_layer_.converged_ is True
        assert model.pattern_layer_.n_iter_ == 7
        np.testing.assert_allclose(model.bandwidth_, expected_bandwidth, rtol=1e-6, atol=1e-7)

    def test_optimize_warns_when_minimize_does_not_converge(self, monkeypatch):
        """При success=False оптимизатор должен сохранить результат и поднять warning."""
        pattern_layer = _make_pattern_layer(
            "per_feature",
            np.array([0.5, 1.5], dtype=np.float64),
            pattern_dtype=np.float64,
        )
        model = DummyGRNNModel(pattern_layer, compute_dtype="float64")
        optimizer = _make_optimizer(
            monkeypatch,
            model,
            lambda y_pred, y_true: 0.0,
            is_pnn_model=False,
        )

        def fake_minimize(fun, theta0, method, bounds, options):
            return SimpleNamespace(
                x=np.array([0.75, 1.25], dtype=np.float64),
                nit=3,
                success=False,
                message="stalled",
            )

        monkeypatch.setattr(optim_module, "minimize", fake_minimize)

        with pytest.warns(RuntimeWarning, match="Optimization did not converge. Reason: stalled"):
            optimizer.optimize()

        np.testing.assert_allclose(optimizer.bandwidth_, np.array([0.75, 1.25], dtype=np.float64))
        assert optimizer.bandwidth_.dtype == np.float64
        assert optimizer.converged_ is False
        assert optimizer.n_iter_ == 3
        assert model.pattern_layer_.converged_ is False
