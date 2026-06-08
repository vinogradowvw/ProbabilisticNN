import numpy as np


def log_likelihood_ratio_loss(y_true, y_pred, f, proba=None, eps=1e-8):
    """Minimize the negative true-vs-competitor log likelihood ratio."""
    y_true = np.asarray(y_true, dtype=np.intp)

    if f.shape[1] < 2:
        return float(f.sum() * 0.0)

    idx = np.arange(f.shape[0], dtype=np.intp)
    f_true = f[idx, y_true]

    competitors = f.copy()
    competitors[idx, y_true] = -np.inf
    f_competitor = np.max(competitors, axis=1)
    log_lr = np.log(f_true + eps) - np.log(f_competitor + eps)
    return float(-np.mean(log_lr))


def correct_class_probability_loss(y_true, y_pred, f, proba, eps=1e-8):
    """Maximize the posterior probability assigned to the correct class."""
    y_true = np.asarray(y_true, dtype=np.intp)
    idx = np.arange(proba.shape[0], dtype=np.intp)
    return float(-np.mean(proba[idx, y_true]))


def _chunked_grad_contract(
    coef,
    X_for_grad,
    W,
    bw_for_kernel,
    kernel,
    bandwidth_sharing,
    normalized,
    pattern_classes,
    n_classes,
):
    """Contract coef with log_kernel_grad in row chunks.

    Avoids materialising a full (n_samples, n_patterns, n_features) tensor for
    per_feature and per_class_per_feature sharing modes. Peak allocation is
    (block, n_patterns, n_features) where block is chosen to stay under 256 MB.
    """
    from probabilisticnn.base.utils import _log_grad_block_size

    n_samples, n_patterns = coef.shape
    n_features = X_for_grad.shape[1]
    block = _log_grad_block_size(n_samples, n_patterns, n_features, itemsize=8)
    coef_f64 = np.asarray(coef, dtype=np.float64)

    if bandwidth_sharing == "per_feature":
        grad = np.zeros(n_features, dtype=np.float64)
        for start in range(0, n_samples, block):
            end = min(start + block, n_samples)
            lg = np.asarray(
                kernel.log_grad(
                    X_for_grad[start:end],
                    W,
                    bandwidth=bw_for_kernel,
                    bandwidth_sharing=bandwidth_sharing,
                    normalized=normalized,
                ),
                dtype=np.float64,
            )
            grad += np.einsum("ij,ijd->d", coef_f64[start:end], lg)
        return grad

    # per_class_per_feature
    per_pattern_feature_grad = np.zeros((n_patterns, n_features), dtype=np.float64)
    for start in range(0, n_samples, block):
        end = min(start + block, n_samples)
        lg = np.asarray(
            kernel.log_grad(
                X_for_grad[start:end],
                W,
                bandwidth=bw_for_kernel,
                bandwidth_sharing=bandwidth_sharing,
                normalized=normalized,
            ),
            dtype=np.float64,
        )
        per_pattern_feature_grad += np.einsum("ij,ijd->jd", coef_f64[start:end], lg)
    grad = np.zeros((n_classes, n_features), dtype=np.float64)
    np.add.at(grad, pattern_classes, per_pattern_feature_grad)
    return grad


class BCELoss:
    def __call__(self, y_true, y_pred, f, proba, eps=1e-8):
        """Binary or multiclass BCE over posterior probabilities."""
        proba = np.clip(np.asarray(proba, dtype=proba.dtype), eps, 1.0 - eps)
        y_true = np.asarray(y_true)

        if proba.shape[1] == 1:
            bce_input = proba[:, 0]
            bce_target = y_true.astype(bce_input.dtype, copy=False)
        elif proba.shape[1] == 2:
            bce_input = proba[:, 1]
            bce_target = y_true.astype(bce_input.dtype, copy=False)
        else:
            bce_input = proba
            bce_target = np.eye(proba.shape[1], dtype=bce_input.dtype)[y_true.astype(np.intp)]

        loss = -(bce_target * np.log(bce_input) + (1.0 - bce_target) * np.log(1.0 - bce_input))
        return float(np.mean(loss))

    def grad(self, X, y, y_pred, model, loo=True, bandwidth=None, K=None, eps=1e-8):
        if K is not None:
            X_for_grad = model.pattern_layer_.patterns_
        elif loo:
            X_for_grad = model.pattern_layer_.patterns_
            K = model.pattern_layer_._loo(bandwidth)
        else:
            X_for_grad = X
            K = model.pattern_layer_.transform(X)
        f = model.summation_layer_.transform(K)
        n_samples, n_classes = f.shape
        idx = np.arange(n_samples, dtype=np.intp)

        _, y_encoded = np.unique(y, return_inverse=True)
        priors = model.output_layer_.prior_

        evidence = f * priors[None, :]
        evidence_sum = evidence.sum(axis=1, keepdims=True)
        evidence_sum = np.maximum(evidence_sum, eps)

        proba = evidence / evidence_sum
        proba = np.clip(proba, eps, 1.0 - eps)

        H = np.zeros_like(proba, dtype=proba.dtype)
        if proba.shape[1] == 1:
            target = y.astype(proba.dtype, copy=False)
            p = proba[:, 0]
            H[:, 0] = (p - target) / (n_samples * p * (1.0 - p))
        elif proba.shape[1] == 2:
            target = y_encoded.astype(proba.dtype, copy=False)
            p = proba[:, 1]
            H[:, 1] = (p - target) / (n_samples * p * (1.0 - p))
        else:
            target = np.eye(proba.shape[1], dtype=proba.dtype)[y_encoded]
            H = (
                -(target / proba)
                + (1.0 - target) / (1.0 - proba)
            ) / (n_samples * proba.shape[1])

        weighted_H = np.sum(H * proba, axis=1, keepdims=True)

        G = (
            priors[None, :]
            / evidence_sum
            * (H - weighted_H)
        )

        pattern_classes = np.asarray(
            model.summation_layer_.y_encoded_,
            dtype=np.intp,
        )

        coef = K * G[:, pattern_classes]

        if loo:
            if K.shape[0] != K.shape[1]:
                raise ValueError("LOO gradient expects square K: n_samples == n_patterns.")

            diag = np.arange(K.shape[0])
            coef[diag, diag] = 0.0

        bw = bandwidth if bandwidth is not None else model.pattern_layer_.bandwidth_
        if hasattr(model.pattern_layer_, "_prepare_bandwidth"):
            bw_for_kernel = model.pattern_layer_._prepare_bandwidth(bw)
        else:
            bw_for_kernel = bw

        bandwidth_sharing = model.pattern_layer_.bandwidth_sharing
        if bandwidth_sharing in ("per_feature", "per_class_per_feature"):
            return _chunked_grad_contract(
                coef=coef,
                X_for_grad=X_for_grad,
                W=model.pattern_layer_.patterns_,
                bw_for_kernel=bw_for_kernel,
                kernel=model.pattern_layer_.kernel_,
                bandwidth_sharing=bandwidth_sharing,
                normalized=model.pattern_layer_.normalize,
                pattern_classes=pattern_classes,
                n_classes=n_classes,
            )

        log_kernel_grad = model.pattern_layer_.kernel_.log_grad(
            X_for_grad,
            model.pattern_layer_.patterns_,
            bandwidth=bw_for_kernel,
            bandwidth_sharing=bandwidth_sharing,
            normalized=model.pattern_layer_.normalize,
        )
        return self._contract_log_kernel_grad(
            coef=coef,
            log_kernel_grad=log_kernel_grad,
            bandwidth_sharing=bandwidth_sharing,
            pattern_classes=pattern_classes,
            n_classes=n_classes,
        )

    def _contract_log_kernel_grad(
        self,
        coef,
        log_kernel_grad,
        bandwidth_sharing,
        pattern_classes,
        n_classes,
    ):
        # coef: (n_samples, n_patterns)
        # log_kernel_grad: (n_samples, n_patterns) for scalar/per_class,
        #                  (n_samples, n_patterns, n_features) for per_feature/per_class_per_feature
        if bandwidth_sharing == "scalar":
            return np.sum(coef * log_kernel_grad)

        if bandwidth_sharing == "per_feature":
            return np.einsum("ij,ijd->d", coef, log_kernel_grad)

        if bandwidth_sharing == "per_class":
            per_pattern_grad = np.sum(coef * log_kernel_grad, axis=0)
            grad = np.zeros(n_classes, coef.dtype)
            np.add.at(grad, pattern_classes, per_pattern_grad)
            return grad

        if bandwidth_sharing == "per_class_per_feature":
            per_pattern_feature_grad = np.einsum("ij,ijd->jd", coef, log_kernel_grad)
            n_features = per_pattern_feature_grad.shape[1]
            grad = np.zeros((n_classes, n_features), dtype=coef.dtype)
            np.add.at(grad, pattern_classes, per_pattern_feature_grad)
            return grad

        raise ValueError(f"Unknown bandwidth_sharing={bandwidth_sharing!r}.")


class CrossEntropyLoss:
    def __call__(self, y_true, y_pred, f, proba, eps=1e-8):
        """Cross entropy over posterior probabilities."""
        proba = np.clip(np.asarray(proba, dtype=proba.dtype), eps, 1.0 - eps)
        y_true = np.asarray(y_true, dtype=np.intp)
        idx = np.arange(proba.shape[0], dtype=np.intp)
        return float(-np.mean(np.log(proba[idx, y_true])))

    def grad(self, X, y, y_pred, model, loo=True, bandwidth=None, K=None, eps=1e-8):
        """Gradient of the cross entropy loss."""
        if K is not None:
            X_for_grad = model.pattern_layer_.patterns_
        elif loo:
            X_for_grad = model.pattern_layer_.patterns_
            K = model.pattern_layer_._loo(bandwidth)
        else:
            X_for_grad = X
            K = model.pattern_layer_.transform(X)

        f = model.summation_layer_.transform(K)

        n_samples = f.shape[0]
        idx = np.arange(n_samples, dtype=np.intp)

        _, y_encoded = np.unique(y, return_inverse=True)

        priors = model.output_layer_.prior_
        if priors.ndim == 0:
            priors = np.full(f.shape[1], float(priors), priors.dtype)

        evidence = f * priors[None, :]
        evidence_sum = evidence.sum(axis=1)
        evidence_sum = np.maximum(evidence_sum, eps)

        f_true = f[idx, y_encoded]
        f_true = np.maximum(f_true, eps)
        G = priors[None, :] / evidence_sum[:, None]
        G[idx, y_encoded] -= 1.0 / f_true
        G /= n_samples
        pattern_classes = np.asarray(
            model.summation_layer_.y_encoded_,
            dtype=np.intp,
        )

        coef = K * G[:, pattern_classes]
        if loo:
            if K.shape[0] != K.shape[1]:
                raise ValueError("LOO gradient expects square K: n_samples == n_patterns.")

            diag = np.arange(K.shape[0])
            coef[diag, diag] = 0.0

        bw = bandwidth if bandwidth is not None else model.pattern_layer_.bandwidth_
        if hasattr(model.pattern_layer_, "_prepare_bandwidth"):
            bw_for_kernel = model.pattern_layer_._prepare_bandwidth(bw)
        else:
            bw_for_kernel = bw

        bandwidth_sharing = model.pattern_layer_.bandwidth_sharing
        if bandwidth_sharing in ("per_feature", "per_class_per_feature"):
            return _chunked_grad_contract(
                coef=coef,
                X_for_grad=X_for_grad,
                W=model.pattern_layer_.patterns_,
                bw_for_kernel=bw_for_kernel,
                kernel=model.pattern_layer_.kernel_,
                bandwidth_sharing=bandwidth_sharing,
                normalized=model.pattern_layer_.normalize,
                pattern_classes=pattern_classes,
                n_classes=f.shape[1],
            )

        log_kernel_grad = model.pattern_layer_.kernel_.log_grad(
            X_for_grad,
            model.pattern_layer_.patterns_,
            bandwidth=bw_for_kernel,
            bandwidth_sharing=bandwidth_sharing,
            normalized=model.pattern_layer_.normalize,
        )
        return self._contract_log_kernel_grad(
            coef=coef,
            log_kernel_grad=log_kernel_grad,
            bandwidth_sharing=bandwidth_sharing,
            pattern_classes=pattern_classes,
            n_classes=f.shape[1],
        )

    def _contract_log_kernel_grad(
        self,
        coef,
        log_kernel_grad,
        bandwidth_sharing,
        pattern_classes,
        n_classes,
    ):
        # coef: (n_samples, n_patterns)
        # log_kernel_grad: (n_samples, n_patterns) for scalar/per_class,
        #                  (n_samples, n_patterns, n_features) for per_feature/per_class_per_feature
        if bandwidth_sharing == "scalar":
            return np.sum(coef * log_kernel_grad)

        if bandwidth_sharing == "per_feature":
            return np.einsum("ij,ijd->d", coef, log_kernel_grad)

        if bandwidth_sharing == "per_class":
            per_pattern_grad = np.sum(coef * log_kernel_grad, axis=0)
            grad = np.zeros(n_classes, dtype=coef.dtype)
            np.add.at(grad, pattern_classes, per_pattern_grad)
            return grad

        if bandwidth_sharing == "per_class_per_feature":
            per_pattern_feature_grad = np.einsum("ij,ijd->jd", coef, log_kernel_grad)
            n_features = per_pattern_feature_grad.shape[1]
            grad = np.zeros((n_classes, n_features), dtype=coef.dtype)
            np.add.at(grad, pattern_classes, per_pattern_feature_grad)
            return grad

        raise ValueError(f"Unknown bandwidth_sharing={bandwidth_sharing!r}.")


class MSELoss:

    def __call__(self, y_pred, y_true):
        """Mean squared error."""
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        return float(np.mean(np.square(y_pred - y_true)))

    def grad(self, X, y, y_pred, model, loo=True, bandwidth=None, K=None):
        """Gradient of the MSE loss."""
        model_grad = model.grad(X, y, bandwidth=bandwidth, loo=loo, K=K)
        upstream = 2.0 * (y_pred - y) / X.shape[0]
        grad_loss = upstream @ model_grad
        return grad_loss


class HuberLoss:
    def __init__(self, delta=1.0):
        self.delta = float(delta)

    def __call__(self, y_pred, y_true):
        """Huber loss."""
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        error = y_pred - y_true
        abs_error = np.abs(error)

        quadratic = 0.5 * np.square(error)
        linear = self.delta * (abs_error - 0.5 * self.delta)
        loss = np.where(abs_error <= self.delta, quadratic, linear)
        return float(np.mean(loss))

    def grad(self, X, y, y_pred, model, loo=True, bandwidth=None, K=None):
        """Gradient of the Huber loss."""
        model_grad = model.grad(X, y, bandwidth=bandwidth, loo=loo, K=K)
        error = np.asarray(y_pred) - np.asarray(y)
        upstream = np.where(
            np.abs(error) <= self.delta,
            error,
            self.delta * np.sign(error),
        ) / X.shape[0]
        grad_loss = upstream @ model_grad
        return grad_loss


class MAELoss:
    def __call__(self, y_pred, y_true):
        """Mean absolute error."""
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        return float(np.mean(np.abs(y_pred - y_true)))


PNN_LOSS_REGISTRY = {
    "log_likelihood_ratio": log_likelihood_ratio_loss,
    "correct_class_probability": correct_class_probability_loss,
    "bce": BCELoss(),
    "cross_entropy": CrossEntropyLoss(),
}

GRNN_LOSS_REGISTRY = {
    "mse": MSELoss(),
    "mae": MAELoss(),
    "huber": HuberLoss(),
}


def _resolve_loss(loss: str, model):
    """Resolve a string loss name into a callable loss function."""
    from probabilisticnn.pnn.pnn import PNN
    from probabilisticnn.grnn.grnn import GRNN

    if isinstance(model, PNN):
        if callable(loss):
            return loss
        try:
            return PNN_LOSS_REGISTRY[loss.lower()]
        except KeyError as exc:
            available = ", ".join(sorted(PNN_LOSS_REGISTRY))
            raise ValueError(f"Unknown loss={loss!r}. Available: {available}") from exc
    if isinstance(model, GRNN):
        if callable(loss):
            return loss
        try:
            return GRNN_LOSS_REGISTRY[loss.lower()]
        except KeyError as exc:
            available = ", ".join(sorted(GRNN_LOSS_REGISTRY))
            raise ValueError(f"Unknown loss={loss!r}. Available: {available}") from exc
