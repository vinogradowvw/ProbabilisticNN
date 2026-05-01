import warnings

import numpy as np
from scipy.optimize import minimize

from base.loss import _resolve_loss
from base.utils import cast_to_dtype


SOLVER_REGISTRY = {
    "auto": "L-BFGS-B",
    "lbfgs": "L-BFGS-B",
    "slsqp": "SLSQP",
    "nelder_mead": "Nelder-Mead",
    "powell": "Powell",
}
INVALID_OBJECTIVE = 1e12


def _resolve_solver(solver):
    try:
        return SOLVER_REGISTRY[solver.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(SOLVER_REGISTRY))
        raise ValueError(f"Unknown solver={solver!r}. Available: {available}") from exc


class BandwidthOptimizer:
    """Optimize AdaptivePNN/AdaptiveGRNN bandwidth parameters.

    Оптимизирует параметры ширины AdaptivePNN/AdaptiveGRNN.
    """

    def __init__(
        self,
        model,
        loss="log_likelihood_ratio",
        max_iter=100,
        solver="auto",
        solver_options=None,
    ):
        self.max_iter = max_iter
        self.loss = loss
        self.loss_fn_ = _resolve_loss(loss, model)
        self.model = model
        self.solver = _resolve_solver(solver)
        self.solver_options = solver_options

    def _is_pnn_model(self):
        if not hasattr(self, "is_pnn_model"):
            from pnn.pnn import PNN
            self.is_pnn_model = isinstance(self.model, PNN)
        return self.is_pnn_model

    def _infer_dtype(self):
        compute_dtype = getattr(self.model, "compute_dtype", "auto")
        if compute_dtype in {"float32", "float64"}:
            return compute_dtype

        pattern_dtype = getattr(self.model.pattern_layer_.patterns_, "dtype", None)
        if pattern_dtype == np.float32:
            return "float32"
        if pattern_dtype == np.float64:
            return "float64"
        return "auto"

    def _forward_state(self, bandwidth):
        """Run the model's LOO training forward pass.

        Выполняет обучающий LOO forward pass модели.
        """
        if self._is_pnn_model():
            y_pred = self.model._forward_train(bandwidth, return_encoded=True)
            f = self.model.summation_layer_.last_f_
            proba = self.model.output_layer_.posteriori(f)
            return y_pred, f, proba

        y_pred = self.model._forward_train(bandwidth)
        return y_pred, None, None

    def _pack_bandwidth(self):
        """Pack current bandwidth parameters into a flat optimization vector.

        Упаковывает текущие параметры ширины в вектор оптимизации.
        """
        bandwidth_params = np.asarray(
            self.model.pattern_layer_.bandwidth_params,
            dtype=np.float64,
        )
        if self.model.pattern_layer_.bandwidth_sharing == "per_class_per_feature":
            return bandwidth_params.ravel().copy()
        return bandwidth_params.copy()

    def _unpack_bandwidth(self, theta):
        """Unpack raw bandwidth parameters from the optimization vector.

        Распаковывает исходные параметры ширины из вектора оптимизации.
        """
        theta = np.asarray(theta, dtype=np.float64)
        pattern_layer = self.model.pattern_layer_
        bandwidth_sharing = pattern_layer.bandwidth_sharing

        if bandwidth_sharing == "per_feature":
            return theta.reshape(pattern_layer.feature_size)
        if bandwidth_sharing == "per_class":
            return theta.reshape(pattern_layer.n_classes_)
        if bandwidth_sharing == "per_class_per_feature":
            return theta.reshape(pattern_layer.n_classes_, pattern_layer.feature_size)

        raise ValueError(f"Unknown bandwidth_sharing={bandwidth_sharing!r}.")

    def _objective(self, theta):
        """Compute the scalar optimization objective.

        Вычисляет скалярное значение целевой функции оптимизации.
        """
        bandwidth = self._unpack_bandwidth(theta)
        if not np.isfinite(bandwidth).all() or np.any(bandwidth <= 0):
            return INVALID_OBJECTIVE

        y_pred, f, proba = self._forward_state(bandwidth)

        if self._is_pnn_model():
            y_true = self.model.output_layer_.y_encoded_
            loss = self.loss_fn_(y_true, y_pred, f, proba)
        else:
            y_true = self.model.summation_layer_.y_
            loss = self.loss_fn_(y_pred, y_true)

        return float(loss) if np.isfinite(loss) else INVALID_OBJECTIVE

    def optimize(self):
        """Optimize bandwidths and store the best observed parameters.

        Оптимизирует ширины и сохраняет лучшие найденные параметры.
        """
        theta0 = np.asarray(self._pack_bandwidth(), dtype=np.float64)
        min_bandwidth = 1e-4
        theta0 = np.maximum(theta0, min_bandwidth)
        bounds = [(min_bandwidth, None)] * theta0.size

        result = minimize(
            self._objective,
            theta0,
            method=self.solver,
            bounds=bounds,
            options={
                "maxiter": self.max_iter,
                **(self.solver_options or {}),
            },
        )

        bandwidth = cast_to_dtype(self._unpack_bandwidth(result.x), self._infer_dtype())

        self.solver_ = self.solver
        self.optimization_result_ = result
        self.bandwidth_ = bandwidth
        self.n_iter_ = getattr(result, "nit", None)
        self.converged_ = bool(result.success)

        self.model.pattern_layer_.bandwidth_ = self.bandwidth_
        self.model.pattern_layer_.bandwidth_params = self.bandwidth_.copy()
        self.model.pattern_layer_.converged_ = self.converged_
        self.model.pattern_layer_.n_iter_ = self.n_iter_

        if hasattr(self.model, "bandwidth_"):
            self.model.bandwidth_ = self.bandwidth_

        if not self.converged_:
            warnings.warn(
                f"Optimization did not converge. Reason: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        return self
