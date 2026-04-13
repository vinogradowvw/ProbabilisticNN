import numpy as np

from collections.abc import Callable

KernelCallable = Callable[[np.ndarray, np.ndarray, float, bool], np.ndarray]