"""Тесты для отдельных layer-компонентов библиотеки."""

import numpy as np
import pytest

from probabilisticnn.base.kernels import gaussian_kernel
from probabilisticnn.base.utils import normalize_l2
from probabilisticnn.common.pattern_layer import AdaptivePatternLayer
from probabilisticnn.common.pattern_layer import PatternLayer
from probabilisticnn.grnn.layers import SummationLayer as GRNNSummationLayer
from probabilisticnn.pnn.layers import OutputLayer
from probabilisticnn.pnn.layers import SummationLayer as PNNSummationLayer
from sklearn.exceptions import NotFittedError


FLOAT_DTYPES = (np.float32, np.float64)


def _compute_dtype_name(dtype) -> str:
    """Возвращает строковое имя dtype в формате, который ожидает библиотека."""
    return np.dtype(dtype).name


class TestPatternLayer:
    """Проверка базового PatternLayer."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_fit_normalizes_patterns_when_requested(self, dtype):
        """При normalize=True слой должен хранить L2-нормализованные паттерны."""
        X = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=dtype)

        layer = PatternLayer(bandwidth=0.5, kernel="gaussian", normalize=True).fit(X)

        np.testing.assert_allclose(layer.patterns_, normalize_l2(X))
        assert layer.kernel_ is gaussian_kernel
        assert layer.bandwidth_ == 0.5
        assert layer.patterns_.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_fit_keeps_raw_patterns_without_normalization(self, dtype):
        """При normalize=False слой не должен менять обучающие паттерны."""
        X = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=dtype)

        layer = PatternLayer(bandwidth=0.5, kernel="gaussian", normalize=False).fit(X)

        np.testing.assert_array_equal(layer.patterns_, X)
        assert layer.patterns_.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_transform_matches_kernel_call(self, dtype):
        """transform должен быть тонкой оберткой над выбранной функцией ядра."""
        X_train = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=dtype)
        X_test = np.array([[0.0, 2.0], [1.0, 1.0]], dtype=dtype)
        layer = PatternLayer(
            bandwidth=np.dtype(dtype).type(0.75),
            kernel="gaussian",
            normalize=True,
        ).fit(X_train)

        actual = layer.transform(X_test)
        expected = gaussian_kernel(
            normalize_l2(X_test),
            normalize_l2(X_train),
            bandwidth=np.dtype(dtype).type(0.75),
            bandwidth_sharing="scalar",
            normalized=True,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        assert actual.dtype == np.dtype(dtype)

    def test_transform_before_fit_raises_not_fitted(self):
        """Слой обязан падать до fit, чтобы не работать с пустым состоянием."""
        layer = PatternLayer(bandwidth=0.5, kernel="gaussian", normalize=True)

        with pytest.raises(NotFittedError):
            layer.transform(np.array([[1.0, 0.0]], dtype=np.float64))

class TestAdaptivePatternLayer:
    """Проверка адаптивного pattern layer и его внутреннего контракта."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_fit_per_feature_initializes_positive_bandwidth_vector(self, dtype):
        """per_feature должен инициализировать по одной ширине на каждый признак."""
        X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=dtype)

        layer = AdaptivePatternLayer(
            kernel="gaussian",
            bandwidth_sharing="per_feature",
            normalize=False,
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X)

        expected = (np.std(X, axis=0) + np.array(1e-12, dtype=dtype)).astype(dtype)

        np.testing.assert_allclose(layer.bandwidth_params, expected, rtol=1e-10, atol=1e-12)
        assert layer.bandwidth_params.shape == (2,)
        assert np.all(layer.bandwidth_params > 0)
        assert layer.bandwidth_params.dtype == np.dtype(dtype)

    def test_fit_per_class_requires_targets(self):
        """Режим per_class не может инициализироваться без целевых меток."""
        X = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)

        layer = AdaptivePatternLayer(
            kernel="gaussian",
            bandwidth_sharing="per_class",
            normalize=False,
        )

        with pytest.raises(ValueError, match="`y` is required"):
            layer.fit(X)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_fit_per_class_initializes_one_bandwidth_per_class(self, dtype):
        """per_class должен получить по одной ширине на каждый класс."""
        X = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [10.0, 4.0],
                [14.0, 8.0],
            ],
            dtype=dtype,
        )
        y = np.array([0, 0, 1, 1])

        layer = AdaptivePatternLayer(
            kernel="gaussian",
            bandwidth_sharing="per_class",
            normalize=False,
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X, y)

        expected = np.array(
            [
                np.std(X[y == 0], axis=0).mean() + 1e-12,
                np.std(X[y == 1], axis=0).mean() + 1e-12,
            ],
            dtype=dtype,
        )

        np.testing.assert_allclose(layer.bandwidth_params, expected, rtol=1e-10, atol=1e-12)
        assert layer.n_classes_ == 2
        assert layer.bandwidth_params.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_fit_per_class_per_feature_initializes_matrix(self, dtype):
        """per_class_per_feature должен хранить матрицу ширин class x feature."""
        X = np.array(
            [
                [0.0, 1.0],
                [2.0, 5.0],
                [10.0, 4.0],
                [14.0, 8.0],
            ],
            dtype=dtype,
        )
        y = np.array([0, 0, 1, 1])

        layer = AdaptivePatternLayer(
            kernel="gaussian",
            bandwidth_sharing="per_class_per_feature",
            normalize=False,
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X, y)

        expected = np.vstack(
            [
                np.std(X[y == 0], axis=0) + 1e-12,
                np.std(X[y == 1], axis=0) + 1e-12,
            ]
        )

        np.testing.assert_allclose(layer.bandwidth_params, expected, rtol=1e-10, atol=1e-12)
        assert layer.bandwidth_params.shape == (2, 2)
        assert layer.bandwidth_params.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_prepare_bandwidth_broadcasts_per_class_vector_to_patterns(self, dtype):
        """Вектор per_class должен раскладываться по обучающим объектам их класса."""
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 4.0], [14.0, 8.0]], dtype=dtype)
        y = np.array([0, 0, 1, 1])
        layer = AdaptivePatternLayer(
            kernel="gaussian",
            bandwidth_sharing="per_class",
            normalize=False,
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X, y)
        raw_bandwidth = np.array([0.5, 2.0], dtype=dtype)

        actual = layer._prepare_bandwidth(raw_bandwidth)
        expected = raw_bandwidth[layer.y_encoded_]

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        assert actual.shape == (X.shape[0],)
        assert actual.dtype == np.dtype(dtype)

    @pytest.mark.parametrize(
        ("sharing", "bandwidth"),
        [
            ("per_feature", np.array([0.5, 1.0, 2.0], dtype=np.float64)),
            ("per_class", np.array([[0.5], [1.0]], dtype=np.float64)),
            ("per_class_per_feature", np.array([0.5, 1.0], dtype=np.float64)),
        ],
    )
    def test_prepare_bandwidth_rejects_invalid_shapes(self, sharing, bandwidth):
        """_prepare_bandwidth должен отсеивать формы, несовместимые с sharing-режимом."""
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 4.0], [14.0, 8.0]], dtype=np.float64)
        y = np.array([0, 0, 1, 1])
        layer = AdaptivePatternLayer(
            kernel="gaussian",
            bandwidth_sharing=sharing,
            normalize=False,
            compute_dtype="float64",
        ).fit(X, None if sharing == "per_feature" else y)

        with pytest.raises(ValueError, match="Invalid bandwidth shape"):
            layer._prepare_bandwidth(bandwidth)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_loo_zeroes_diagonal(self, dtype):
        """Leave-one-out kernel matrix обязана занулять диагональ similarity."""
        X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]], dtype=dtype)
        layer = AdaptivePatternLayer(
            kernel="gaussian",
            bandwidth_sharing="per_feature",
            normalize=False,
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X)

        actual = layer._loo()

        assert actual.shape == (3, 3)
        np.testing.assert_allclose(np.diag(actual), np.zeros(3), atol=1e-12)
        assert actual.dtype == np.dtype(dtype)

    def test_transform_requires_optimized_state(self):
        """transform доступен только после того, как оптимизатор выставил bandwidth_."""
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        layer = AdaptivePatternLayer(
            kernel="gaussian",
            bandwidth_sharing="per_feature",
            normalize=False,
            compute_dtype="float64",
        ).fit(X)

        with pytest.raises(NotFittedError):
            layer.transform(X)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_transform_matches_kernel_after_bandwidth_is_set(self, dtype):
        """После оптимизации адаптивный слой должен вызывать ядро как обычный layer."""
        X_train = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
        X_test = np.array([[1.0, 1.0]], dtype=dtype)
        layer = AdaptivePatternLayer(
            kernel="gaussian",
            bandwidth_sharing="per_feature",
            normalize=False,
            compute_dtype=_compute_dtype_name(dtype),
        ).fit(X_train)
        layer.bandwidth_ = layer.bandwidth_params.copy()
        layer.converged_ = True

        actual = layer.transform(X_test)
        expected = gaussian_kernel(
            X_test,
            X_train,
            bandwidth=layer.bandwidth_params,
            bandwidth_sharing="per_feature",
            normalized=False,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        assert actual.dtype == np.dtype(dtype)

class TestPNNSummationLayer:
    """Проверка слоя суммирования для PNN."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_transform_aggregates_kernel_mass_by_class(self, dtype):
        """Слой должен суммировать плотность по объектам одного класса."""
        X = np.array([[0.0], [1.0], [2.0]], dtype=dtype)
        y = np.array([0, 1, 1])
        K = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=dtype,
        )
        layer = PNNSummationLayer().fit(X, y)

        actual = layer.transform(K)
        expected = np.array(
            [
                [1.0, 5.0],
                [4.0, 11.0],
            ],
            dtype=dtype,
        ) / 2.0

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(layer.last_f_, expected, rtol=1e-10, atol=1e-12)
        assert actual.dtype == np.dtype(dtype)
        assert layer.last_f_.dtype == np.dtype(dtype)

    def test_transform_rejects_wrong_number_of_columns(self):
        """Число колонок K должно совпадать с числом обучающих паттернов."""
        X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
        y = np.array([0, 1, 1])
        layer = PNNSummationLayer().fit(X, y)

        with pytest.raises(ValueError, match="expected 3 train samples"):
            layer.transform(np.array([[1.0, 2.0]], dtype=np.float64))

    def test_transform_before_fit_raises_not_fitted(self):
        """Слой не должен принимать kernel матрицу до fit."""
        layer = PNNSummationLayer()

        with pytest.raises(NotFittedError):
            layer.transform(np.array([[1.0, 2.0]], dtype=np.float64))

class TestOutputLayer:
    """Проверка выходного слоя PNN."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_fit_uniform_losses_uses_float_compute_dtype(self, dtype):
        """prior и likelihood multiplier должны быть в вычислительном float dtype."""
        y = np.array([1, 1, 0, 1])

        layer = OutputLayer(losses="uniform", compute_dtype=_compute_dtype_name(dtype)).fit(y)

        np.testing.assert_allclose(layer.prior_, np.array([0.25, 0.75], dtype=dtype))
        np.testing.assert_allclose(
            layer.likelihood_multiplier_,
            np.array([0.25, 0.75], dtype=dtype),
        )
        assert layer.prior_.dtype == np.dtype(dtype)
        assert layer.likelihood_multiplier_.dtype == np.dtype(dtype)

    def test_transform_encoded_and_transform_apply_class_weights_and_restore_labels(self):
        """Слой должен и выбирать закодированный класс, и восстанавливать исходные метки."""
        y = np.array(["class_b", "class_a", "class_b"])
        f = np.array(
            [
                [0.2, 0.4],
                [0.05, 0.9],
            ],
            dtype=np.float64,
        )
        layer = OutputLayer(
            losses=np.array([4.0, 0.5], dtype=np.float64),
            compute_dtype="float64",
        ).fit(y)

        encoded = layer.transform_encoded(f)
        labels = layer.transform(f)

        np.testing.assert_array_equal(encoded, np.array([0, 1]))
        np.testing.assert_array_equal(labels, np.array(["class_a", "class_b"]))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_posteriori_returns_prior_for_zero_density_rows(self, dtype):
        """При нулевой суммарной плотности posterior должен откатываться к prior."""
        y = np.array([0, 1, 1])
        f = np.array(
            [
                [0.0, 0.0],
                [1.0, 3.0],
            ],
            dtype=dtype,
        )
        layer = OutputLayer(losses="uniform", compute_dtype=_compute_dtype_name(dtype)).fit(y)

        actual = layer.posteriori(f)
        expected = np.array(
            [
                [1.0 / 3.0, 2.0 / 3.0],
                [1.0 / 7.0, 6.0 / 7.0],
            ],
            dtype=dtype,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        assert actual.dtype == np.dtype(dtype)

    def test_fit_rejects_custom_losses_with_wrong_shape(self):
        """Кастомный вектор loss должен содержать ровно одно значение на класс."""
        y = np.array([0, 1, 1])
        layer = OutputLayer(losses=np.array([1.0, 2.0, 3.0]), compute_dtype="float64")

        with pytest.raises(ValueError, match="one value per class"):
            layer.fit(y)


class TestGRNNSummationLayer:
    """Проверка слоя суммирования для GRNN."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_transform_returns_kernel_weighted_average(self, dtype):
        """GRNN summation layer должен считать взвешенное среднее по similarity."""
        y = np.array([1.0, 2.0, 4.0], dtype=dtype)
        K = np.array(
            [
                [1.0, 1.0, 2.0],
                [0.0, 3.0, 1.0],
            ],
            dtype=dtype,
        )
        layer = GRNNSummationLayer().fit(y)

        actual = layer.transform(K)
        expected = np.array(
            [
                (1.0 * 1.0 + 1.0 * 2.0 + 2.0 * 4.0) / 4.0,
                (0.0 * 1.0 + 3.0 * 2.0 + 1.0 * 4.0) / 4.0,
            ],
            dtype=dtype,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        assert actual.dtype == np.dtype(dtype)

    def test_transform_returns_zero_for_zero_similarity_rows(self):
        """Если строка similarity полностью нулевая, регрессионный ответ должен быть нулем."""
        y = np.array([1.0, 2.0], dtype=np.float32)
        K = np.array([[0.0, 0.0], [1.0, 3.0]], dtype=np.float32)
        layer = GRNNSummationLayer().fit(y)

        actual = layer.transform(K)

        np.testing.assert_allclose(actual, np.array([0.0, 1.75], dtype=np.float32), atol=1e-6)
        assert actual.dtype == np.float32
