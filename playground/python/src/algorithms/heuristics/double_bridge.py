from typing import Tuple

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def double_bridge(tour: np.ndarray, matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """ Двойной мост - непоследовательный 4-opt
    tour: список городов
    matrix: матрица весов
    return: выигрыш, новый тур
    """
    best_gain, exchange, size = 0.0, (0, 0, 0, 0), len(tour)

    for x in range(size):
        for y in range(x + 1, size):
            for z in range(y + 1, size):
                for w in range(z + 1, size):
                    if (w + 1) % size == x:
                        continue
                    it11, it12 = tour[x % size], tour[(x + 1) % size]
                    it21, it22 = tour[y % size], tour[(y + 1) % size]
                    it31, it32 = tour[z % size], tour[(z + 1) % size]
                    it41, it42 = tour[w % size], tour[(w + 1) % size]
                    old = matrix[it11][it12] + matrix[it21][it22] + matrix[it31][it32] + matrix[it41][it42]
                    new = matrix[it11][it32] + matrix[it12][it31] + matrix[it21][it42] + matrix[it22][it41]
                    gain = old - new
                    if best_gain < gain:
                        best_gain, exchange = gain, (x, y, z, w)
                        break

    if best_gain > 1.e-10:
        x, y, z, w = exchange
        idx = 0
        temp = np.zeros(size, dtype=nb.int64)
        idx = __copy_slice(temp, tour, x + 1, y, idx)
        idx = __copy_slice(temp, tour, w + 1, x, idx)
        idx = __copy_slice(temp, tour, z + 1, w, idx)
        __copy_slice(temp, tour, y + 1, z, idx)
        return best_gain, temp

    return 0., tour


@nb.njit(cache=True)
def __copy_slice(temp: np.ndarray, tour: np.ndarray, x: int, y: int, idx: int) -> int:
    """ Перекопирование тура в правильном порудке для двойного моста
    temp: куда копируем
    tour: текущий тур
    x, y, idx: откуда, докуда, куда
    """
    size = len(tour)
    i, j = x % size, (y + 1) % size
    while i != j:
        temp[idx] = tour[i]
        i = (i + 1) % size
        idx += 1
    return idx
