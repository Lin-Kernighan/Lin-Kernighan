from typing import Tuple

import numba as nb
import numpy as np

from src.algorithms.utils.abc_opt import AbcOpt
from src.algorithms.utils.utils import swap


@nb.njit(cache=True)
def _search(matrix: np.ndarray, tour: np.ndarray, x: int, y: int, z: int) -> Tuple[int, float]:
    """ Поиск лучшей замены, среди всех возможных замен
    matrix: Матрица весов
    tour: Список городов
    x, y, z: Города, для которых пробуем найти изменение тура
    return: Тип переворота, выигрыш
    """
    s = len(tour)
    a, b, c, d, e, f = tour[x % s], tour[(x + 1) % s], tour[y % s], tour[(y + 1) % s], tour[z % s], tour[(z + 1) % s]
    base = current_min = matrix[a][b] + matrix[c][d] + matrix[e][f]
    gain, exchange = 0, -1

    if current_min > (current := matrix[a][e] + matrix[c][d] + matrix[b][f]):  # 2-opt (a, e) (d, c) (b, f)
        gain, exchange, current_min = base - current, 0, current
    if current_min > (current := matrix[a][b] + matrix[c][e] + matrix[d][f]):  # 2-opt (a, b) (c, e) (d, f)
        gain, exchange, current_min = base - current, 1, current
    if current_min > (current := matrix[a][c] + matrix[b][d] + matrix[e][f]):  # 2-opt (a, c) (b, d) (e, f)
        gain, exchange, current_min = base - current, 2, current
    if current_min > (current := matrix[a][d] + matrix[e][c] + matrix[b][f]):  # 3-opt (a, d) (e, c) (b, f)
        gain, exchange, current_min = base - current, 3, current
    if current_min > (current := matrix[a][d] + matrix[e][b] + matrix[c][f]):  # 3-opt (a, d) (e, b) (c, f)
        gain, exchange, current_min = base - current, 4, current
    if current_min > (current := matrix[a][e] + matrix[d][b] + matrix[c][f]):  # 3-opt (a, e) (d, b) (c, f)
        gain, exchange, current_min = base - current, 5, current
    if current_min > (current := matrix[a][c] + matrix[b][e] + matrix[d][f]):  # 3-opt (a, c) (b, e) (d, f)
        gain, exchange, current_min = base - current, 6, current

    return exchange, gain


@nb.njit(cache=True)
def _exchange(tour: np.ndarray, best_exchange: int, nodes: tuple) -> np.ndarray:
    """ Изменение тура, после нахождения лучшего изменения 3-opt
    tour: Список городов
    best_exchange: Тип замены
    nodes: Города
    return: Новый список городов
    """
    x, y, z = nodes
    s = len(tour)
    a, b, c, d, e, f = x % s, (x + 1) % s, y % s, (y + 1) % s, z % s, (z + 1) % s
    if best_exchange == 0:
        tour = swap(tour, b, e)
    elif best_exchange == 1:
        tour = swap(tour, d, e)
    elif best_exchange == 2:
        tour = swap(tour, b, c)
    elif best_exchange == 3:
        tour = swap(swap(tour, b, e), b, b + (e - d))
    elif best_exchange == 4:
        tour = swap(swap(swap(tour, b, e), b, b + (e - d)), e - (c - b), e)
    elif best_exchange == 5:
        tour = swap(swap(tour, b, e), e - (c - b), e)
    elif best_exchange == 6:
        tour = swap(swap(tour, d, e), b, c)
    return tour


class ThreeOpt(AbcOpt):
    """ Локальный поиск: 3-opt
    Ищем три ребра, которые можно перецепить, чтобы уменьшить длину тура.
    Продолжаем до тех пор, пока есть такой случай.
    Вычислительная сложность поиска локального минимума: O(n^3)
    """

    def __init__(self, length: float, tour: np.ndarray, matrix: np.ndarray, **kwargs):
        super().__init__(length, tour, matrix, **kwargs)

    def improve(self, **kwargs) -> float:
        """ Локальный поиск (поиск изменения + само изменение)
        return: выигрыш от локального поиска
        """
        best_exchange, best_gain, best_nodes = self._improve(self.matrix, self.tour)
        if best_gain > 1.e-10:
            self.tour = _exchange(self.tour, best_exchange, best_nodes)
            self.length -= best_gain
            self.collector.update({'length': self.length, 'gain': best_gain})
            return best_gain
        return 0.0

    @staticmethod
    @nb.njit(cache=True)
    def _improve(matrix: np.ndarray, tour: np.ndarray) -> Tuple[int, float, tuple]:
        """ Основной цикл 3-opt: поиск лучшего измения тура
        matrix: Матрица весов
        tour: Список городов
        return: Тип переворота, выигрыш, переворачиваемый интервал
        """
        best_exchange, best_gain, best_nodes = 0, 0, (0, 0, 0)
        size = matrix.shape[0]

        for x in range(size):
            for y in range(x + 1, size):
                for z in range(y + 1, size):
                    exchange, gain = _search(matrix, tour, x, y, z)
                    if gain > best_gain:
                        best_gain, best_exchange, best_nodes = gain, exchange, (x, y, z)

        return best_exchange, best_gain, best_nodes
