from typing import Tuple

import numba as nb
import numpy as np

from src.algorithms.utils.abc_opt import AbcOpt
from src.algorithms.utils.utils import swap


class TwoOpt(AbcOpt):
    """ Локальный поиск: 2-opt
    Ищем два ребра, которые можно перецепить, чтобы уменьшить длину тура.
    Продолжаем до тех пор, пока есть такое преобразование.
    Вычислительная сложность поиска локального минимума: O(n^2)
    length: начальная длина тура
    tour: начальный тур
    matrix: матрица весов
    """

    def __init__(self, length: float, tour: np.ndarray, matrix: np.ndarray, **kwargs):
        super().__init__(length, tour, matrix, **kwargs)

    def improve(self) -> float:
        """ Локальный поиск (поиск изменения + само изменение)
        return: выигрыш от локального поиска
        """
        saved, best_change = self._improve(self.matrix, self.tour)
        if best_change < 0:
            i, j = saved
            self.tour = swap(self.tour, i + 1, j)
            self.length += best_change
            self.collector.update({'length': self.length, 'gain': -best_change})
            return -best_change
        return 0.0

    @staticmethod
    @nb.njit(cache=True)
    def _improve(matrix: np.ndarray, tour: np.ndarray) -> Tuple[tuple, float]:
        """ Основной цикл 2-opt: поиск лучшего измения тура
        matrix: Матрица весов
        tour: Список городов
        return: переворачиваемый интервал, выигрыш
        """
        best_change, saved = 0, (0, 0)
        size = matrix.shape[0]

        for n in range(size):
            for m in range(n + 1, size):
                i, j = tour[n % size], tour[m % size]
                x, y = tour[(n + 1) % size], tour[(m + 1) % size]
                change = matrix[i][j] + matrix[x][y]
                change -= matrix[i][x] + matrix[j][y]
                if change < best_change:
                    best_change = change
                    saved = (n, m)

        return saved, best_change
