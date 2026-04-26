import numpy as np

class SummationLayer:

    def fit(self, y):
        self.y_ = y
        return self
    
    def transform(self, K):
        denom = np.sum(K, axis=1)
        nom = np.matmul(K, self.y_)
        out = np.divide(nom, denom, out=np.zeros_like(nom, dtype=np.float64), where=denom > 0)
        return out
