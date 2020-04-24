from sys import maxsize

import numba as nb
import numpy as np

from src.utils import rotate_zero


def get_hash(tour: np.ndarray) -> int:
    return hash(str(tour))


@nb.experimental.jitclass(spec=[
    ('data', nb.types.Set(nb.i8)),
    ('best_length', nb.float64),
    ('best_route', nb.int64[:])
])
class TabuSet:

    def __init__(self):
        self.data = {1, 0}
        self.data.clear()
        self.best_length = maxsize
        self.best_route = np.array([1] * 1, dtype=np.int64)

    def is_contains(self, item: np.ndarray) -> bool:
        with nb.objmode(x='intp'):
            x = get_hash(item)
        if x in self.data:
            return True
        return False

    def __add(self, item: np.ndarray) -> None:
        with nb.objmode(x='intp'):
            x = get_hash(item)
        self.data.add(x)

    def append(self, tour: np.ndarray, length: float) -> bool:
        with nb.objmode(tour='intp[:]'):
            tour = rotate_zero(tour)
        if self.is_contains(tour):
            return False
        self.__add(tour)
        if length < self.best_length:
            self.best_route, self.best_length = tour.copy(), length
        return True

    def best_result(self) -> float:
        return self.best_length

    def best_tour(self) -> np.ndarray:
        return self.best_route.copy()
