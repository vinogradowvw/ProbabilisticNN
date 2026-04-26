import numpy as np


def log_likelihood_ratio_loss(y_true, y_pred, f, proba=None, eps=1e-8):
    """Minimize the negative true-vs-competitor log likelihood ratio.

    Минимизирует отрицательное логарифмическое отношение правдоподобия
    истинного класса к ближайшему конкурирующему классу.
    """
    f = np.asarray(f, dtype=np.float64)
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
    """Maximize the posterior probability assigned to the correct class.

    Максимизирует апостериорную вероятность, назначенную правильному классу.
    """
    proba = np.asarray(proba, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.intp)
    idx = np.arange(proba.shape[0], dtype=np.intp)
    return float(-np.mean(proba[idx, y_true]))


def bce_loss(y_true, y_pred, f, proba, eps=1e-8):
    """Binary or multiclass BCE over posterior probabilities.

    Binary или multiclass BCE по апостериорным вероятностям.
    """
    proba = np.clip(np.asarray(proba, dtype=np.float64), eps, 1.0 - eps)
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


def cross_entropy_loss(y_true, y_pred, f, proba, eps=1e-8):
    """Cross entropy over posterior probabilities.

    Cross entropy по апостериорным вероятностям.
    """
    proba = np.clip(np.asarray(proba, dtype=np.float64), eps, 1.0 - eps)
    y_true = np.asarray(y_true, dtype=np.intp)
    idx = np.arange(proba.shape[0], dtype=np.intp)
    return float(-np.mean(np.log(proba[idx, y_true])))


def mse_loss(y_pred, y_true):
    """Mean squared error.

    Среднеквадратичная ошибка.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.mean(np.square(y_pred - y_true)))


def mae_loss(y_pred, y_true):
    """Mean absolute error.

    Средняя абсолютная ошибка.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.mean(np.abs(y_pred - y_true)))


PNN_LOSS_REGISTRY = {
    "log_likelihood_ratio": log_likelihood_ratio_loss,
    "correct_class_probability": correct_class_probability_loss,
    "bce": bce_loss,
    "cross_entropy": cross_entropy_loss,
}

GRNN_LOSS_REGISTRY = {
    "mse": mse_loss,
    "mae": mae_loss,
}


def _resolve_loss(loss: str, model):
    """Resolve a string loss name into a callable loss function.

    Преобразует строковое имя loss-функции в вызываемую функцию.
    """
    from pnn.pnn import PNN
    from grnn.grnn import GRNN

    if isinstance(model, PNN):
        try:
            return PNN_LOSS_REGISTRY[loss.lower()]
        except KeyError as exc:
            available = ", ".join(sorted(PNN_LOSS_REGISTRY))
            raise ValueError(f"Unknown loss={loss!r}. Available: {available}") from exc
    if isinstance(model, GRNN):
        try:
            return GRNN_LOSS_REGISTRY[loss.lower()]
        except KeyError as exc:
            available = ", ".join(sorted(GRNN_LOSS_REGISTRY))
            raise ValueError(f"Unknown loss={loss!r}. Available: {available}") from exc

