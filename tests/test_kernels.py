import numpy as np
import pytest

from probabilisticnn.base.kernels import exponential_kernel
from probabilisticnn.base.kernels import gaussian_kernel
from probabilisticnn.base.kernels import laplacian_kernel
from probabilisticnn.base.kernels import resolve_kernel


KERNELS = {
    "gaussian": gaussian_kernel,
    "laplacian": laplacian_kernel,
    "exponential": exponential_kernel,
}
FLOAT_DTYPES = (np.float32, np.float64)


def _make_bandwidth(sharing: str, dtype=np.float64):
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


def _reference_kernel(kernel_name: str, X, W, bandwidth, sharing: str):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    out = np.empty((X.shape[0], W.shape[0]), dtype=np.float64)

    for i, x in enumerate(X):
        for j, w in enumerate(W):
            if sharing == "scalar":
                local_bandwidth = float(bandwidth)
            elif sharing == "per_feature":
                local_bandwidth = np.asarray(bandwidth, dtype=np.float64)
            elif sharing == "per_class":
                local_bandwidth = float(np.asarray(bandwidth, dtype=np.float64)[j])
            elif sharing == "per_class_per_feature":
                local_bandwidth = np.asarray(bandwidth, dtype=np.float64)[j]
            else:
                raise ValueError(f"Unsupported sharing mode for test helper: {sharing!r}")

            diff = x - w
            if kernel_name == "gaussian":
                if sharing in {"scalar", "per_class"}:
                    scaled_distance = np.dot(diff, diff) / (2.0 * (local_bandwidth ** 2))
                else:
                    scaled_distance = np.sum((diff ** 2) / (2.0 * (local_bandwidth ** 2)))
            elif kernel_name == "laplacian":
                if sharing in {"scalar", "per_class"}:
                    scaled_distance = np.abs(diff).sum() / local_bandwidth
                else:
                    scaled_distance = np.sum(np.abs(diff) / local_bandwidth)
            elif kernel_name == "exponential":
                if sharing in {"scalar", "per_class"}:
                    scaled_distance = np.linalg.norm(diff) / local_bandwidth
                else:
                    scaled_distance = np.sqrt(np.sum((diff / local_bandwidth) ** 2))
            else:
                raise ValueError(f"Unsupported kernel for test helper: {kernel_name!r}")

            out[i, j] = np.exp(-max(float(scaled_distance), 0.0))

    return out


def _assert_close_for_dtype(actual, expected, dtype):
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


class TestKernelMathematicalProperties:
    @pytest.mark.parametrize("kernel_name", list(KERNELS))
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
    def test_matches_reference_implementation(self, kernel_name, dtype, sharing):
        kernel = KERNELS[kernel_name]
        X = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=dtype)
        W = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0]], dtype=dtype)
        bandwidth = _make_bandwidth(sharing, dtype=dtype)

        expected = _reference_kernel(kernel_name, X, W, bandwidth, sharing)
        actual = kernel(X, W, bandwidth, sharing, normalized=False)

        _assert_close_for_dtype(actual, expected, dtype)
        assert actual.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("kernel_name", list(KERNELS))
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
    def test_identical_vectors_produce_unit_similarity(self, kernel_name, dtype, sharing):
        kernel = KERNELS[kernel_name]
        X = np.array([[0.0, 1.0], [1.5, -0.5], [-1.0, 2.0]], dtype=dtype)
        bandwidth = _make_bandwidth(sharing, dtype=dtype)

        actual = kernel(X, X, bandwidth, sharing, normalized=False)

        np.testing.assert_allclose(np.diag(actual), np.ones(X.shape[0]), atol=1e-12)
        assert actual.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("kernel_name", list(KERNELS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("sharing", ["scalar", "per_feature"])
    def test_shared_bandwidth_kernels_are_symmetric_on_self_comparison(self, kernel_name, dtype, sharing):
        kernel = KERNELS[kernel_name]
        X = np.array([[0.0, 1.0], [1.5, -0.5], [-1.0, 2.0]], dtype=dtype)
        bandwidth = _make_bandwidth(sharing, dtype=dtype)

        actual = kernel(X, X, bandwidth, sharing, normalized=False)

        np.testing.assert_allclose(actual, actual.T, rtol=1e-10, atol=1e-12)
        assert actual.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("kernel_name", list(KERNELS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_similarity_decreases_with_distance_for_scalar_bandwidth(self, kernel_name, dtype):
        kernel = KERNELS[kernel_name]
        X = np.array([[0.0, 0.0]], dtype=dtype)
        W = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=dtype)

        actual = kernel(X, W, 1.0, "scalar", normalized=False)[0]

        assert actual[0] > actual[1] > actual[2]
        assert actual.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("kernel_name", ["gaussian", "exponential"])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_normalized_scalar_matches_euclidean_formula_on_unit_vectors(self, kernel_name, dtype):
        kernel = KERNELS[kernel_name]
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)
        W = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=dtype)
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        bandwidth = np.dtype(dtype).type(0.8)

        normalized = kernel(X, W, bandwidth, "scalar", normalized=True)
        euclidean = kernel(X, W, bandwidth, "scalar", normalized=False)

        _assert_close_for_dtype(normalized, euclidean, dtype)
        assert normalized.dtype == np.dtype(dtype)
        assert euclidean.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("kernel_name", list(KERNELS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_per_class_per_feature_uses_its_own_bandwidth_row_for_three_classes(self, kernel_name, dtype):
        kernel = KERNELS[kernel_name]

        # One sample compared against three class prototypes with identical geometry.
        # The only thing that changes across columns is the per-class-per-feature bandwidth.
        X = np.array([[1.0, 2.0]], dtype=dtype)
        W = np.zeros((3, 2), dtype=dtype)
        bandwidth = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [0.5, 0.5],
            ],
            dtype=dtype,
        )

        actual = kernel(X, W, bandwidth, "per_class_per_feature", normalized=False)
        expected = _reference_kernel(
            kernel_name,
            X,
            W,
            bandwidth,
            "per_class_per_feature",
        )

        _assert_close_for_dtype(actual, expected, dtype)
        assert actual[0, 1] > actual[0, 0] > actual[0, 2]
        assert actual.dtype == np.dtype(dtype)


class TestKernelApiContract:
    @pytest.mark.parametrize("name", ["gaussian", "GAUSSIAN", "Gaussian"])
    def test_resolve_kernel_is_case_insensitive(self, name):
        assert resolve_kernel(name) is gaussian_kernel

    def test_resolve_kernel_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            resolve_kernel("unknown")

    @pytest.mark.parametrize("kernel_name", list(KERNELS))
    @pytest.mark.parametrize(
        "bandwidth",
        [
            0.0,
            -1.0,
            np.array([0.5, 0.0]),
            np.array([0.5, -1.0]),
        ],
    )
    def test_rejects_non_positive_bandwidth(self, kernel_name, bandwidth):
        kernel = KERNELS[kernel_name]
        X = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float64)
        W = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="strictly positive"):
            kernel(X, W, bandwidth, "scalar", normalized=False)

    @pytest.mark.parametrize("kernel_name", list(KERNELS))
    def test_rejects_unknown_bandwidth_sharing(self, kernel_name):
        kernel = KERNELS[kernel_name]
        X = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float64)
        W = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="Unknown bandwidth_sharing"):
            kernel(X, W, 0.75, "unsupported", normalized=False)

    @pytest.mark.parametrize("kernel_name", list(KERNELS))
    @pytest.mark.parametrize(
        ("sharing", "bandwidth"),
        [
            ("scalar", np.array([0.5, 1.0], dtype=np.float64)),
            ("per_feature", np.array([0.5, 1.0, 1.5], dtype=np.float64)),
            ("per_class", np.array([0.5, 1.0], dtype=np.float64)),
            (
                "per_class_per_feature",
                np.array([[0.5, 1.0], [1.0, 0.5]], dtype=np.float64),
            ),
        ],
    )
    def test_invalid_bandwidth_shapes_raise(self, kernel_name, sharing, bandwidth):
        kernel = KERNELS[kernel_name]
        X = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float64)
        W = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0]], dtype=np.float64)

        with pytest.raises(ValueError):
            kernel(X, W, bandwidth, sharing, normalized=False)
