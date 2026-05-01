import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data, check_is_fitted


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
        self.class_mask_ = np.zeros((n_train, self.n_classes_), dtype=X.dtype)
        self.class_mask_[np.arange(n_train), self.y_encoded_] = 1.0
        return self

    def transform(self, K):
        check_is_fitted(self, ["classes_", "class_mask_", "n_train_"])

        if K.shape[1] != self.n_train_:
            raise ValueError(
                f"K has {K.shape[1]} columns, but expected {self.n_train_} train samples."
            )

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
        self.prior_ = np.asarray((counts / counts.sum()), dtype=y.dtype)

        if self.losses == "uniform":
            losses = np.ones(self.n_classes_, dtype=y.dtype)
        else:
            losses = np.asarray(self.losses, dtype=y.dtype)
            if losses.shape != (self.n_classes_,):
                raise ValueError("`losses` must have one value per class.")

        self.likelihood_multiplier_ = self.prior_ * losses
        return self

    def transform_encoded(self, f):
        """Predict encoded class indices from class densities.

        Предсказывает закодированные индексы классов по значениям плотности.
        """
        likelihood_multiplier = np.asarray(self.likelihood_multiplier_, dtype=f.dtype)
        posterior = f * likelihood_multiplier
        return np.argmax(posterior, axis=1)

    def transform(self, f):
        classes_enc = self.transform_encoded(f)
        classes = self.classes_[classes_enc]
        return classes

    def posteriori(self, f):
        prior = np.asarray(self.prior_, dtype=f.dtype)
        nom = prior * f
        denom = np.sum(nom, axis=1, keepdims=True)
        fallback = np.broadcast_to(prior, nom.shape).copy()
        return np.divide(nom, denom, out=fallback, where=denom > 0)
