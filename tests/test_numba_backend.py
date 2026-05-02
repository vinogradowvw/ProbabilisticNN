"""Тесты parity и API-контракта для numba backend."""

import importlib.util

import numpy as np
import pytest


if importlib.util.find_spec("numba") is None:
    pytest.skip("numba is required for numba_backend tests", allow_module_level=True)


from probabilisticnn.base.kernels import exponential_kernel as numpy_exponential_kernel
from probabilisticnn.base.kernels import gaussian_kernel as numpy_gaussian_kernel
from probabilisticnn.base.kernels import laplacian_kernel as numpy_laplacian_kernel
from probabilisticnn.grnn.layers import SummationLayer as GRNNSummationLayer
from probabilisticnn.numba_backend import grnn_jit_inference
from probabilisticnn.numba_backend import pnn_jit_inference
from probabilisticnn.numba_backend.kernels import exponential_kernel as numba_exponential_kernel
from probabilisticnn.numba_backend.kernels import gaussian_kernel as numba_gaussian_kernel
from probabilisticnn.numba_backend.kernels import laplacian_kernel as numba_laplacian_kernel
from probabilisticnn.numba_backend.kernels import resolve_kernel as resolve_numba_kernel


NUMPY_KERNELS = {
    "gaussian": numpy_gaussian_kernel,
    "laplacian": numpy_laplacian_kernel,
    "exponential": numpy_exponential_kernel,
}

NUMBA_KERNELS = {
    "gaussian": numba_gaussian_kernel,
    "laplacian": numba_laplacian_kernel,
    "exponential": numba_exponential_kernel,
}

FLOAT_DTYPES = (np.float32, np.float64)


def _make_bandwidth(sharing: str, dtype=np.float64):
    """Возвращает bandwidth в форме, соответствующей режиму sharing."""
    dtype = np.dtype(dtype)

    if sharing == "scalar":
        return dtype.type(0.75)
    if sharing == "per_feature":
        return np.array([0.5, 1.25], dtype=dtype)
    if sharing == "per_class":
        return np.array([0.5, 1.0, 1.5], dtype=dtype)
    if sharing == "per_class_per_feature":
        return np.array(
            [
                [0.5, 1.25],
                [1.0, 0.75],
                [1.5, 0.6],
            ],
            dtype=dtype,
        )
    raise ValueError(f"Unsupported sharing mode for test helper: {sharing!r}")


def _assert_close_for_dtype(actual, expected, dtype):
    """Подбирает допустимую ошибку с учетом более грубого float32."""
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def _make_kernel_inputs(dtype, sharing, *, normalized=False):
    """Готовит детерминированные входы для проверки порядка компиляции и dtype."""
    if normalized:
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)
        W = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=dtype)
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
    else:
        X = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=dtype)
        W = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0]], dtype=dtype)

    bandwidth = _make_bandwidth(sharing, dtype=dtype)
    return X, W, bandwidth


def _manual_pnn_inference(
    kernel_name,
    X,
    W,
    y_encoded,
    n_classes,
    likelihood_multiplier,
    bandwidth,
    bandwidth_sharing,
    normalized=False,
):
    """Эталонная NumPy-реализация PNN inference для проверки numba-ветки."""
    kernel = NUMPY_KERNELS[kernel_name]
    K = kernel(X, W, bandwidth, bandwidth_sharing, normalized)

    # В PNN суммирование идет по классам, поэтому вручную строим class mask.
    class_mask = np.zeros((y_encoded.shape[0], n_classes), dtype=X.dtype)
    class_mask[np.arange(y_encoded.shape[0]), y_encoded] = 1.0
    f = np.dot(K, class_mask) / n_classes
    posterior = f * np.asarray(likelihood_multiplier, dtype=X.dtype)
    return np.argmax(posterior, axis=1)


def _manual_grnn_inference(
    kernel_name,
    X,
    W,
    y,
    bandwidth,
    bandwidth_sharing,
    normalized=False,
):
    """Эталонная NumPy-реализация GRNN inference для проверки numba-ветки."""
    kernel = NUMPY_KERNELS[kernel_name]
    K = kernel(X, W, bandwidth, bandwidth_sharing, normalized)
    return GRNNSummationLayer().fit(y).transform(K)


class TestNumbaKernels:
    """Сравнение numba kernels с базовой NumPy-реализацией."""

    @pytest.mark.parametrize("kernel_name", list(NUMBA_KERNELS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize(
        "sharing",
        [
            "scalar",
            "per_feature",
            "per_class",
            "per_class_per_feature",
        ],
    )
    def test_matches_numpy_backend(self, kernel_name, dtype, sharing):
        """Numba kernels должны численно совпадать с numpy backend-ом."""
        numpy_kernel = NUMPY_KERNELS[kernel_name]
        numba_kernel = NUMBA_KERNELS[kernel_name]
        X = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=dtype)
        W = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0]], dtype=dtype)
        bandwidth = _make_bandwidth(sharing, dtype=dtype)

        expected = numpy_kernel(X, W, bandwidth, sharing, normalized=False)
        actual = numba_kernel(X, W, bandwidth, sharing, normalized=False)

        _assert_close_for_dtype(actual, expected, dtype)
        assert actual.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("kernel_name", ["gaussian", "exponential"])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_normalized_scalar_matches_numpy_backend(self, kernel_name, dtype):
        """ветка scalar normalized должна совпадать с NumPy и по числам, и по dtype."""
        numpy_kernel = NUMPY_KERNELS[kernel_name]
        numba_kernel = NUMBA_KERNELS[kernel_name]
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)
        W = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=dtype)
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        bandwidth = np.dtype(dtype).type(0.8)

        expected = numpy_kernel(X, W, bandwidth, "scalar", normalized=True)
        actual = numba_kernel(X, W, bandwidth, "scalar", normalized=True)

        _assert_close_for_dtype(actual, expected, dtype)
        assert actual.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("kernel_name", list(NUMBA_KERNELS))
    @pytest.mark.parametrize("sharing", ["scalar", "per_feature", "per_class", "per_class_per_feature"])
    def test_accepts_float32_and_float64_independently_of_call_order(self, kernel_name, sharing):
        """Первый вызов на одном dtype не должен ломаться для другого dtype."""
        numpy_kernel = NUMPY_KERNELS[kernel_name]
        numba_kernel = NUMBA_KERNELS[kernel_name]

        for dtype_order in ((np.float32, np.float64), (np.float64, np.float32)):
            results = []

            for dtype in dtype_order:
                X, W, bandwidth = _make_kernel_inputs(dtype, sharing, normalized=False)
                expected = numpy_kernel(X, W, bandwidth, sharing, normalized=False)
                actual = numba_kernel(X, W, bandwidth, sharing, normalized=False)

                _assert_close_for_dtype(actual, expected, dtype)
                assert actual.dtype == np.dtype(dtype)
                results.append(actual)

            assert results[0].dtype == np.dtype(dtype_order[0])
            assert results[1].dtype == np.dtype(dtype_order[1])

    def test_resolve_kernel_rejects_unknown_name(self):
        """Неизвестное имя ядра должно отбрасываться явно."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            resolve_numba_kernel("unknown")

    @pytest.mark.parametrize("kernel_name", list(NUMBA_KERNELS))
    def test_rejects_non_positive_bandwidth(self, kernel_name):
        """Numba backend должен валидировать bandwidth так же, как и NumPy backend."""
        kernel = NUMBA_KERNELS[kernel_name]
        X = np.array([[0.0, 1.0]], dtype=np.float64)
        W = np.array([[0.0, 1.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="strictly positive"):
            kernel(X, W, 0.0, "scalar", normalized=False)

    @pytest.mark.parametrize("kernel_name", list(NUMBA_KERNELS))
    def test_rejects_unknown_bandwidth_sharing(self, kernel_name):
        """Публичный API numba ядер должен защищаться от неизвестного sharing-режима."""
        kernel = NUMBA_KERNELS[kernel_name]
        X = np.array([[0.0, 1.0]], dtype=np.float64)
        W = np.array([[0.0, 1.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="Unknown bandwidth_sharing"):
            kernel(X, W, 0.75, "unsupported", normalized=False)


class TestNumbaInference:
    """Сравнение numba inference с ручным NumPy-путем."""

    @pytest.mark.parametrize("kernel_name", list(NUMBA_KERNELS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize(
        "sharing",
        [
            "scalar",
            "per_feature",
            "per_class",
            "per_class_per_feature",
        ],
    )
    def test_pnn_jit_inference_matches_numpy_path(self, kernel_name, dtype, sharing):
        """PNN numba inference должен выбирать те же классы, что и эталонный путь."""
        X = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=dtype)
        W = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0]], dtype=dtype)
        y_encoded = np.array([0, 1, 1], dtype=np.intp)
        likelihood_multiplier = np.array([1.0, 0.6], dtype=dtype)
        bandwidth = _make_bandwidth(sharing, dtype=dtype)

        expected = _manual_pnn_inference(
            kernel_name,
            X,
            W,
            y_encoded,
            n_classes=2,
            likelihood_multiplier=likelihood_multiplier,
            bandwidth=bandwidth,
            bandwidth_sharing=sharing,
            normalized=False,
        )
        actual = pnn_jit_inference(
            kernel=kernel_name,
            X=X,
            W=W,
            y_encoded=y_encoded,
            n_classes=2,
            likelihood_multiplier=likelihood_multiplier,
            bandwidth=bandwidth,
            bandwidth_sharing=sharing,
            normalized=False,
        )

        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype.kind in {"i", "u"}

    @pytest.mark.parametrize("kernel_name", list(NUMBA_KERNELS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize(
        "sharing",
        [
            "scalar",
            "per_feature",
            "per_class",
            "per_class_per_feature",
        ],
    )
    def test_grnn_jit_inference_matches_numpy_path_and_preserves_dtype(self, kernel_name, dtype, sharing):
        """GRNN numba inference должен совпадать с NumPy и сохранять вычислительный dtype."""
        X = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=dtype)
        W = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0]], dtype=dtype)
        y = np.array([1.0, 2.0, 4.0], dtype=dtype)
        bandwidth = _make_bandwidth(sharing, dtype=dtype)

        expected = _manual_grnn_inference(
            kernel_name,
            X,
            W,
            y,
            bandwidth=bandwidth,
            bandwidth_sharing=sharing,
            normalized=False,
        )
        actual = grnn_jit_inference(
            kernel=kernel_name,
            X=X,
            W=W,
            y=y,
            bandwidth=bandwidth,
            bandwidth_sharing=sharing,
            normalized=False,
        )

        _assert_close_for_dtype(actual, expected, dtype)
        assert actual.dtype == np.dtype(dtype)

    def test_inference_accepts_float32_and_float64_independently_of_call_order(self):
        """JIT helper-ы должны корректно переживать смену dtype между вызовами."""
        X_base = np.array([[0.0, 1.0], [2.0, -1.0]])
        W_base = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0]])
        y_encoded = np.array([0, 1, 1], dtype=np.intp)

        for dtype_order in ((np.float32, np.float64), (np.float64, np.float32)):
            pnn_results = []
            grnn_results = []

            for dtype in dtype_order:
                X = X_base.astype(dtype, copy=True)
                W = W_base.astype(dtype, copy=True)
                bandwidth = np.dtype(dtype).type(0.75)
                likelihood_multiplier = np.array([1.0, 0.6], dtype=dtype)
                y_reg = np.array([1.0, 2.0, 4.0], dtype=dtype)

                expected_pnn = _manual_pnn_inference(
                    "gaussian",
                    X,
                    W,
                    y_encoded,
                    n_classes=2,
                    likelihood_multiplier=likelihood_multiplier,
                    bandwidth=bandwidth,
                    bandwidth_sharing="scalar",
                    normalized=False,
                )
                actual_pnn = pnn_jit_inference(
                    kernel="gaussian",
                    X=X,
                    W=W,
                    y_encoded=y_encoded,
                    n_classes=2,
                    likelihood_multiplier=likelihood_multiplier,
                    bandwidth=bandwidth,
                    bandwidth_sharing="scalar",
                    normalized=False,
                )

                expected_grnn = _manual_grnn_inference(
                    "gaussian",
                    X,
                    W,
                    y_reg,
                    bandwidth=bandwidth,
                    bandwidth_sharing="scalar",
                    normalized=False,
                )
                actual_grnn = grnn_jit_inference(
                    kernel="gaussian",
                    X=X,
                    W=W,
                    y=y_reg,
                    bandwidth=bandwidth,
                    bandwidth_sharing="scalar",
                    normalized=False,
                )

                np.testing.assert_array_equal(actual_pnn, expected_pnn)
                _assert_close_for_dtype(actual_grnn, expected_grnn, dtype)
                assert actual_grnn.dtype == np.dtype(dtype)

                pnn_results.append(actual_pnn)
                grnn_results.append(actual_grnn)

            assert pnn_results[0].dtype.kind in {"i", "u"}
            assert pnn_results[1].dtype.kind in {"i", "u"}
            assert grnn_results[0].dtype == np.dtype(dtype_order[0])
            assert grnn_results[1].dtype == np.dtype(dtype_order[1])
