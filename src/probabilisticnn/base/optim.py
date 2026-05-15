import warnings

import numpy as np
from scipy.optimize import minimize

from probabilisticnn.base.utils import cast_to_dtype
from probabilisticnn.base.loss import _resolve_loss


SOLVER_REGISTRY = {
    "auto": "SLSQP",
    "lbfgs": "L-BFGS-B",
    "l_bfgs_b": "L-BFGS-B",
    "bfgs": "BFGS",
    "cg": "CG",
    "newton_cg": "Newton-CG",
    "slsqp": "SLSQP",
    "tnc": "TNC",
    "nelder_mead": "Nelder-Mead",
    "powell": "Powell",
    "cobyla": "COBYLA",
    "cobyqa": "COBYQA",
}

GRADIENT_SOLVERS = {
    "L-BFGS-B",
    "BFGS",
    "CG",
    "Newton-CG",
    "SLSQP",
    "TNC",
}
BOUNDED_SOLVERS = {
    "L-BFGS-B",
    "SLSQP",
    "TNC",
    "Powell",
    "Nelder-Mead",
    "COBYQA",
}
DEFAULT_MIN_BANDWIDTH = 1e-6
DEFAULT_MAX_BANDWIDTH = 1e6
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
            from probabilisticnn.pnn.pnn import PNN
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

    def _bandwidth_limits(self):
        """Return safe lower/upper bounds for bandwidth parameters."""
        min_bandwidth = float(getattr(self.model, "min_bandwidth", DEFAULT_MIN_BANDWIDTH))
        max_bandwidth = float(getattr(self.model, "max_bandwidth", DEFAULT_MAX_BANDWIDTH))

        if not np.isfinite(min_bandwidth) or not np.isfinite(max_bandwidth):
            raise ValueError("Bandwidth bounds must be finite.")
        if min_bandwidth <= 0:
            raise ValueError("min_bandwidth must be strictly positive.")
        if min_bandwidth >= max_bandwidth:
            raise ValueError("min_bandwidth must be smaller than max_bandwidth.")

        return min_bandwidth, max_bandwidth

    def _theta_limits(self):
        """Return bandwidth bounds mapped into the log-parameterization."""
        min_bandwidth, max_bandwidth = self._bandwidth_limits()
        return np.log(min_bandwidth), np.log(max_bandwidth)

    def _theta_is_safe(self, theta):
        """Return whether theta is finite and within the solver bounds."""
        theta = np.asarray(theta, dtype=np.float64)
        if not np.isfinite(theta).all():
            return False

        theta_min, theta_max = self._theta_limits()
        return bool(np.all(theta >= theta_min) and np.all(theta <= theta_max))

    def _pack_bandwidth(self):
        """Pack current bandwidth parameters into a flat optimization vector.

        Упаковывает текущие параметры ширины в вектор оптимизации.

        Оптимизуем логарифмы чтобы обойтись без оптимизации с ограничеями
        """
        min_bandwidth, max_bandwidth = self._bandwidth_limits()
        bandwidth_params = np.asarray(
            np.clip(self.model.pattern_layer_.bandwidth_params, min_bandwidth, max_bandwidth),
            dtype=np.float64,
        )
        bandwidth_params = np.log(bandwidth_params)
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
            return np.exp(theta).reshape(pattern_layer.feature_size)
        if bandwidth_sharing == "per_class":
            return np.exp(theta).reshape(pattern_layer.n_classes_)
        if bandwidth_sharing == "per_class_per_feature":
            return np.exp(theta).reshape(pattern_layer.n_classes_, pattern_layer.feature_size)

        raise ValueError(f"Unknown bandwidth_sharing={bandwidth_sharing!r}.")

    def _objective(self, theta, *args, **kwargs):
        """Compute the scalar optimization objective.

        Вычисляет скалярное значение целевой функции оптимизации.
        """
        theta = np.asarray(theta, dtype=np.float64)
        if not self._theta_is_safe(theta):
            return INVALID_OBJECTIVE

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

    def _jac(self, theta, X, y, model, loo=True):
        """Compute the gradient of the objective function.
        """
        theta = np.asarray(theta, dtype=np.float64)
        if not self._theta_is_safe(theta):
            return np.zeros_like(theta)

        bandwidth = self._unpack_bandwidth(theta)
        if not np.isfinite(bandwidth).all() or np.any(bandwidth <= 0):
            return np.zeros_like(theta)

        y_pred, f, proba = self._forward_state(bandwidth)
        grad_bandwidth = np.asarray(
            self.loss_fn_.grad(X, y, y_pred, model, loo=loo, bandwidth=bandwidth),
            dtype=np.float64,
        )

        # Optimization runs in theta = log(bandwidth), so apply
        # dL/dtheta = dL/dbandwidth * dbandwidth/dtheta = dL/dbandwidth * bandwidth.
        grad_theta = grad_bandwidth * np.asarray(bandwidth, dtype=np.float64)
        return np.ravel(grad_theta)

    def optimize(self):
        """Optimize bandwidths and store the best observed parameters.

        Оптимизирует ширины и сохраняет лучшие найденные параметры.
        """
        theta_min, theta_max = self._theta_limits()
        theta0 = np.asarray(self._pack_bandwidth(), dtype=np.float64)
        theta0 = np.clip(theta0, theta_min, theta_max)

        uses_gradient = self.solver in GRADIENT_SOLVERS and hasattr(self.loss_fn_, "grad")
        minimize_kwargs = {}
        if self.solver in BOUNDED_SOLVERS:
            minimize_kwargs["bounds"] = [(theta_min, theta_max)] * theta0.size

        result = minimize(
            self._objective,
            theta0,
            method=self.solver,
            jac=self._jac if uses_gradient else None,
            args=(
                self.model.pattern_layer_.patterns_,
                self.model.summation_layer_.y_,
                self.model,
                True,
            ),
            options={
                "maxiter": self.max_iter,
                **(self.solver_options or {}),
            },
            **minimize_kwargs,
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
            diagnostic_fields = {
                "solver": self.solver,
                "status": getattr(result, "status", None),
                "success": getattr(result, "success", None),
                "message": getattr(result, "message", None),
                "fun": getattr(result, "fun", None),
                "nit": getattr(result, "nit", None),
                "nfev": getattr(result, "nfev", None),
                "njev": getattr(result, "njev", None),
                "x": getattr(result, "x", None),
                "theta0": theta0,
                "bandwidth": bandwidth,
                "max_iter": self.max_iter,
                "solver_options": self.solver_options,
            }
            diagnostics = ", ".join(
                f"{key}={value!r}" for key, value in diagnostic_fields.items()
            )
            warnings.warn(
                f"Optimization did not converge. Diagnostics: {diagnostics}",
                RuntimeWarning,
                stacklevel=2,
            )

        return self
