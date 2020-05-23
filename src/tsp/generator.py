from math import sqrt

import numpy as np
from numba import njit


@njit(cache=True)
def generator(count: int) -> np.ndarray:
    """ Генерируем случайную TSP — задачу. """
    return np.random.random_sample((count, 2)) * sqrt(count) * 100
