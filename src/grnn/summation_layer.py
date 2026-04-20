import numpy as np
import torch


class SummationLayer:

    def fit(self, y):
        self.y_ = y
        self.y_t_ = torch.as_tensor(y, dtype=torch.float32)
        return self
    
    def transform(self, K):
        if torch.is_tensor(K):
            y = self.y_t_.to(device=K.device, dtype=K.dtype)
            denom = torch.sum(K, dim=1)
            nom = torch.matmul(K, y)
            out = nom / denom
            return out

        denom = np.sum(K, axis=1)
        nom = np.matmul(K, self.y_)
        out = nom / denom
        return out
