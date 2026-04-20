from numpy import isin
import torch
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss


def log_likelihood_ratio_loss(y_true, y_pred, f, proba=None, eps=1e-8):
    """Minimize the negative true-vs-competitor log likelihood ratio.

    Минимизирует отрицательное логарифмическое отношение правдоподобия
    истинного класса к ближайшему конкурирующему классу.
    """
    if f.shape[1] < 2:
        return f.sum() * 0.0

    # Select each sample's true-class density estimate.
    # Выбираем оценку плотности истинного класса для каждого объекта.
    idx = torch.arange(f.size(0), device=f.device)
    f_true = f[idx, y_true]

    # Mask the true class to compare only against competing classes.
    # Маскируем истинный класс, чтобы сравнивать только с классами-конкурентами.
    competitors = f.clone()
    competitors[idx, y_true] = -torch.inf
    f_competitor = torch.max(competitors, dim=1).values
    log_lr = torch.log(f_true + eps) - torch.log(f_competitor + eps)
    return -log_lr.mean()


def correct_class_probability_loss(y_true, y_pred, f, proba, eps=1e-8):
    """Maximize the posterior probability assigned to the correct class.

    Максимизирует апостериорную вероятность, назначенную правильному классу.
    """
    idx = torch.arange(proba.size(0), device=proba.device)
    return -torch.mean(proba[idx, y_true])


def bce_loss(y_true, y_pred, f, proba, eps=1e-8):
    """Binary or multilabel BCE over posterior probabilities.

    Binary или multilabel BCE по апостериорным вероятностям.
    """
    proba = torch.clamp(proba, min=eps, max=1.0 - eps)

    if proba.shape[1] == 1:
        bce_input = proba[:, 0]
        bce_target = y_true.to(dtype=bce_input.dtype)
    elif proba.shape[1] == 2:
        bce_input = proba[:, 1]
        bce_target = y_true.to(dtype=bce_input.dtype)
    else:
        bce_input = proba
        bce_target = F.one_hot(y_true, num_classes=proba.shape[1]).to(dtype=bce_input.dtype)
    return F.binary_cross_entropy(bce_input, bce_target)


def cross_entropy_loss(y_true, y_pred, f, proba, eps=1e-8):
    """Cross entropy over log posterior probabilities.

    Cross entropy по логарифмам апостериорных вероятностей.
    """
    proba = torch.clamp(proba, min=eps, max=1.0 - eps)
    return F.cross_entropy(torch.log(proba), y_true)


PNN_LOSS_REGISTRY = {
    "log_likelihood_ratio": log_likelihood_ratio_loss,
    "correct_class_probability": correct_class_probability_loss,
    "bce": bce_loss,
    "cross_entropy": cross_entropy_loss,
}

GRNN_LOSS_REGISTRY = {
    "mse": MSELoss(),
    "mae": L1Loss()
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


class BandwidthOptimizer:
    """Optimize AdaptivePNN bandwidth parameters with gradient descent.

    Оптимизирует параметры ширины AdaptivePNN с помощью градиентного спуска.
    """

    def __init__(
        self,
        model,
        loss="log_likelihood_ratio",
        lr=1e-2,
        max_iter=100,
        tol=1e-4,
        min_bandwidth=1e-6,
        eps=1e-12,
        verbose=False,
    ):
        self.max_iter = max_iter
        self.loss = loss
        self.loss_fn_ = _resolve_loss(loss, model)
        self.tol = tol
        self.lr = lr
        self.model = model
        self.min_bandwidth = min_bandwidth
        self.eps = eps
        self.verbose = verbose

    def _forward_state(self):
        """Run the model's LOO training forward pass.

        Выполняет обучающий LOO forward pass модели.
        """
        from pnn.pnn import PNN
        y_pred = self.model._forward_train()
        if isinstance(self.model, PNN):
            f = self.model.summation_layer_.last_f_
            proba = self.model.output_layer_.posteriori(f)
            return y_pred, f, proba

        return y_pred, None, None

    def optimize(self):
        """Optimize bandwidths and store the best observed parameters.

        Оптимизирует ширины и сохраняет лучшие найденные параметры.
        """
        from pnn.pnn import PNN
        optimizer = torch.optim.Adam(
            [self.model.pattern_layer_.bandwidth_params],
            lr=self.lr
        )

        device = self.model.pattern_layer_.patterns_t_.device
        if hasattr(self.model.summation_layer_, "y_encoded_"):
            y_true = torch.as_tensor(
                self.model.summation_layer_.y_encoded_,
                dtype=torch.long,
                device=device,
            )
        else:
            y_true = torch.as_tensor(
                self.model.summation_layer_.y_,
                dtype=torch.float32,
                device=device,
            )

        self.loss_history_ = []
        self.relative_change_history_ = []
        self.converged_ = False
        best_loss = float("inf")
        best_weights = self.model.pattern_layer_.bandwidth_params.detach().clone()

        for iteration in range(1, self.max_iter + 1):
            previous_weights = self.model.pattern_layer_.bandwidth_params.detach().clone()
            optimizer.zero_grad()

            y_pred, f, proba = self._forward_state()
            if isinstance(self.model, PNN):
                loss_value = self.loss_fn_(y_true, y_pred, f, proba, self.eps)
            else:
                loss_value = self.loss_fn_(y_pred, y_true)
            loss_value.backward()
            optimizer.step()

            with torch.no_grad():
                # keep bandwidths positive after the optimizer update.
                # cохраняем ширины положительными после шага оптимизатора.
                self.model.pattern_layer_.bandwidth_params.clamp_(min=self.min_bandwidth)
                current_weights = self.model.pattern_layer_.bandwidth_params.detach()
                relative_change = torch.linalg.vector_norm(current_weights - previous_weights) / (
                    torch.linalg.vector_norm(previous_weights) + self.eps
                )

            current_loss = float(loss_value.detach().cpu())
            current_relative_change = float(relative_change.cpu())
            self.loss_history_.append(current_loss)
            self.relative_change_history_.append(current_relative_change)
            if current_loss < best_loss:
                best_loss = current_loss
                best_weights = current_weights.clone()

            if current_relative_change < self.tol:
                self.converged_ = True
                break

            if self.verbose:
                print("Iteration {}: loss={:.4f}, relative_change={:.4f}".format(
                    iteration, current_loss, current_relative_change
                ))

        self.n_iter_ = len(self.loss_history_)
        self.best_loss_ = best_loss
        self.bandwidth_ = best_weights.detach().cpu().numpy().copy()
        self.relative_change_ = self.relative_change_history_[-1]
        with torch.no_grad():
            self.model.pattern_layer_.bandwidth_params.copy_(best_weights)

        self.model.pattern_layer_.bandwidth_ = self.bandwidth_
        self.model.pattern_layer_.converged_ = self.converged_
        self.model.pattern_layer_.n_iter_ = self.n_iter_
        self.model.pattern_layer_.relative_change_ = self.relative_change_

        return self
