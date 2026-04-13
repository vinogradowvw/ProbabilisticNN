import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data, check_is_fitted
import torch


class SummationLayer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_, self.y_encoded_, self.counts_ = np.unique(y, return_inverse=True, return_counts=True)

        n_train = X.shape[0]
        self.n_classes_ = len(self.classes_)
        self.n_train_ = n_train

        # class_mask[i, j] = 1, если i-й обучающий объект принадлежит j-му классу.
        self.class_mask_ = np.zeros((n_train, self.n_classes_), dtype=float)
        self.class_mask_[np.arange(n_train), self.y_encoded_] = 1.0

        self.class_mask_t_ = torch.as_tensor(self.class_mask_, dtype=torch.float32)
        return self

    def transform(self, K):
        check_is_fitted(self, ["classes_", "class_mask_", "n_train_"])

        if K.shape[1] != self.n_train_:
            raise ValueError(
                f"K has {K.shape[1]} columns, but expected {self.n_train_} train samples."
            )

        if torch.is_tensor(K):
            f_unnormalized = torch.matmul(K, self.class_mask_t_)
        else:
            f_unnormalized = np.matmul(K, self.class_mask_)

        f = f_unnormalized / self.n_classes_
        self.last_f_ = f
        return f


class OutputLayer:
    def __init__(self, losses):
        self.losses = losses

    def fit(self, y):
        self.classes_, self.y_encoded_, counts = np.unique(y, return_inverse=True, return_counts=True)
        self.n_classes_ = len(self.classes_)
        self.prior_ = counts / counts.sum()
        self.prior_t_ = torch.as_tensor(self.prior_, dtype=torch.float32)

        if self.losses == "uniform":
            losses = np.ones(self.n_classes_, dtype=float)
        else:
            losses = np.asarray(self.losses, dtype=float)
            if losses.shape != (self.n_classes_,):
                raise ValueError("`losses` must have one value per class.")

        self.likelihood_multiplier_ = self.prior_ * losses
        self.likelihood_multiplier_t_ = torch.as_tensor(self.likelihood_multiplier_, dtype=torch.float32)
        return self

    def transform(self, f):
        if torch.is_tensor(f):
            posterior = f * self.likelihood_multiplier_t_
            classes_enc = torch.argmax(posterior, dim=1)
        else:
            posterior = f * self.likelihood_multiplier_
            classes_enc = np.argmax(posterior, axis=1)
        classes = self.classes_[classes_enc]
        return classes

    def posteriori(self, f):
        if torch.is_tensor(f):
            nom = self.prior_t_.to(f.device) * f
            denom = torch.sum(nom, dim=1, keepdim=True)
        else:
            nom = self.prior_ * f
            denom = np.sum(nom, axis=1, keepdims=True)
        return nom / denom
