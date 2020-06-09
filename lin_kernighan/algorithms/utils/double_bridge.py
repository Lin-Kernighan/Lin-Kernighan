from typing import Tuple

import numba as nb
import numpy as np


@nb.njit(cache=True)
def double_bridge(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray, fast: bool) -> Tuple[float, np.ndarray]:
    """ Двойной мост - непоследовательный 4-opt, вариант с передачей кандидатов значительно быстрее
    tour: список городов
    matrix: матрица весов
    candidates: матрица кандидатов, если не будет использоваться candidates = матрица [2, 2] (для numba)
    fast: возвращать первое же возможное решение
    return: выигрыш, новый тур
    """
    if len(candidates) == 2:
        best_gain, exchange = __find_full(tour, matrix, fast)
    else:
        best_gain, exchange = __find_neighbours(tour, matrix, candidates, fast)

    if best_gain > 1.e-10:
        x, y, z, w = exchange
        idx = 0
        temp = np.zeros(len(tour), dtype=nb.int64)
        idx = __copy_slice(temp, tour, x + 1, y, idx)
        idx = __copy_slice(temp, tour, w + 1, x, idx)
        idx = __copy_slice(temp, tour, z + 1, w, idx)
        __copy_slice(temp, tour, y + 1, z, idx)
        return best_gain, temp

    return 0., tour


@nb.njit(cache=True)
def __find_full(tour: np.ndarray, matrix: np.ndarray, fast: bool) -> Tuple[float, tuple]:
    """ 4-opt по всем возможны вершинам, эффективно, но медленно
    tour: список городов
    matrix: матрица смежности
    fast: возвращать первое же возможное решение
    return: выигрыш, города, на которых выполнять 4-opt
    """
    best_gain, exchange, size = 0.0, (0, 0, 0, 0), len(tour)
    for x in range(size):
        for y in range(x + 1, size):
            for z in range(y + 1, size):
                for w in range(z + 1, size):
                    gain = __get_gain(size, tour, matrix, x, y, z, w)
                    if fast and gain > 1.e-10:
                        return gain, (x, y, z, w)
                    elif best_gain < gain:
                        best_gain, exchange = gain, (x, y, z, w)
    return best_gain, exchange


@nb.njit(cache=True)
def __find_neighbours(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray, fast: bool) -> Tuple[float, tuple]:
    """ 4-opt только по указанным кандидатам
    tour: список городов
    matrix: матрица смежности
    fast: возвращать первое же возможное решение
    return: выигрыш, города, на которых выполнять 4-opt
    """
    best_gain, exchange, size = 0.0, (0, 0, 0, 0), len(tour)
    for ix in range(size):
        x = tour[ix]
        for y in candidates[x]:
            if y == -1:
                continue
            for z in candidates[y]:
                if z == x or z == -1:
                    continue
                for w in candidates[z]:
                    if w == y or w == x or w == -1:
                        continue
                    iy, iz, iw = __get_indexes(tour, y, z, w)
                    if not ix < iy < iz < iw:
                        continue
                    gain = __get_gain(size, tour, matrix, ix, iy, iz, iw)
                    if fast and gain > 1.e-10:
                        return gain, (ix, iy, iz, iw)
                    elif best_gain < gain:
                        best_gain, exchange = gain, (ix, iy, iz, iw)
    return best_gain, exchange


@nb.njit(cache=True)
def __get_indexes(tour: np.ndarray, y: int, z: int, w: int) -> Tuple[int, int, int]:
    """ Получаем индексы для указанных вершин
    tour: список городов
    y, z, w: значения городов
    return: их индексы
    """
    iy, iz, iw, i = -1, -1, -1, 0
    for idx, val in enumerate(tour):
        if y == val:
            iy = idx
            i += 1
        elif z == val:
            iz = idx
            i += 1
        elif w == val:
            iw = idx
            i += 1
        if i == 3:
            break
    return iy, iz, iw


@nb.njit(cache=True)
def __get_gain(size: int, tour: np.ndarray, matrix: np.ndarray, x: int, y: int, z: int, w: int) -> float:
    """ Расчеты выигрыша при замене ребер
    tour: список городов
    size: длина маршрута
    matrix: матрица весов
    x, y, z, w: индексы! городов
    return: выигрыш
    """
    it11, it12 = tour[x % size], tour[(x + 1) % size]
    it21, it22 = tour[y % size], tour[(y + 1) % size]
    it31, it32 = tour[z % size], tour[(z + 1) % size]
    it41, it42 = tour[w % size], tour[(w + 1) % size]
    old = matrix[it11][it12] + matrix[it21][it22] + matrix[it31][it32] + matrix[it41][it42]
    new = matrix[it11][it32] + matrix[it12][it31] + matrix[it21][it42] + matrix[it22][it41]
    return old - new


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
