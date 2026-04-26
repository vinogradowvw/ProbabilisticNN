import numpy as np

class SummationLayer:

    def fit(self, y):
        self.y_ = y
        return self
    
    def transform(self, K):
        denom = np.sum(K, axis=1)
        nom = np.matmul(K, self.y_)
        out = nom / denom
        return out
