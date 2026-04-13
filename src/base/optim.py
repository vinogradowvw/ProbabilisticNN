import torch


def log_likelihood_ratio_loss(
        y_true,  # (n,1)
        y_pred,  # (n,1)
        f,       # (n, n) (must be loo - diagonal is 0)
        eps=1e-8
    ):
    if f.shape[1] < 2:
        return f.sum() * 0.0

    idx = torch.arange(f.size(0), device=f.device)
    f_true = f[idx, y_true]

    # Compare each sample's true-class response against the strongest
    # competing class (not against predicted class, which can equal true class).
    competitors = f.clone()
    competitors[idx, y_true] = -torch.inf
    f_competitor = torch.max(competitors, dim=1).values

    log_lr = torch.log(f_true + eps) - torch.log(f_competitor + eps)
    return -log_lr.mean()


LOSS_REGISTRY = {
    "log_likelihood_ratio": log_likelihood_ratio_loss
}

def _resolve_loss(loss: str):
    try:
        return LOSS_REGISTRY[loss.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss={loss!r}. Available: {available}") from exc


class BandwidthOptimizer:

    def __init__(
        self,
        model,
        loss=log_likelihood_ratio_loss,
        lr=1e-2,
        max_iter=100,
        tol=1e-4,
        min_bandwidth=1e-6,
        eps=1e-12,
        verbose=False,
    ):
        self.max_iter = max_iter
        self.loss = loss
        if isinstance(loss, str):
            self.loss_fn_ = _resolve_loss(loss)
        else:
            self.loss_fn_ = loss
        self.tol = tol
        self.lr = lr
        self.model = model
        self.min_bandwidth = min_bandwidth
        self.eps = eps
        self.verbose = verbose

    def optimize(self):
        optimizer = torch.optim.Adam(
            [self.model.pattern_layer_.bandwidth_params],
            lr=self.lr
        )

        device = self.model.pattern_layer_.patterns_t_.device

        y_true = torch.as_tensor(self.model.summation_layer_.y_encoded_, dtype=torch.long, device=device)
        class_mask = torch.as_tensor(
            self.model.summation_layer_.class_mask_,
            dtype=torch.float32,
            device=device,
        )
        n_classes = self.model.summation_layer_.n_classes_

        self.loss_history_ = []
        self.relative_change_history_ = []
        self.converged_ = False
        best_loss = float("inf")
        best_weights = self.model.pattern_layer_.bandwidth_params.detach().clone()

        for iteration in range(1, self.max_iter + 1):
            previous_weights = self.model.pattern_layer_.bandwidth_params.detach().clone()
            optimizer.zero_grad()

            K_loo = self.model.pattern_layer_._loo()
            f_unnormalized = torch.matmul(K_loo, class_mask)
            f = f_unnormalized / n_classes
            y_pred = torch.argmax(f, dim=1)
            self.model.summation_layer_.last_f = f

            loss_value = self.loss_fn_(y_true, y_pred, f)
            loss_value.backward()
            optimizer.step()

            with torch.no_grad():
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
