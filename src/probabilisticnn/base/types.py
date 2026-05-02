import numpy as np

from collections.abc import Callable


KernelCallable = Callable[
    [
        np.ndarray,  # X
        np.ndarray,  # W
        np.ndarray,  # bandwidth
        str,  # bandwidth_sharing 
        bool  # normalized
    ],
    np.ndarray
]
