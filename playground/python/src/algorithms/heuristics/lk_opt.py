import logging
from typing import Tuple

import numba as nb
import numpy as np

from src.algorithms.heuristics.abc_opt import AbcOpt
from src.utils import make_pair, swap, between, around


@nb.njit(cache=True)
def _get_tour(tour: np.ndarray, it1: int, it2: int, it3: int, it4: int) -> np.ndarray:
    """ Выполняем k-opt для указанных города
    tour: список городов
    it1, it2, it3, it4: найденные города
    return: новый тур
    """
    if it1 < it2 < it4:
        return swap(tour, it2, it4)  # ... it1 it2 ... it4 it3 ...
    elif it4 < it3 < it1:
        return swap(tour, it3, it1)  # ... it4 it3 ... it1 it2 ...
    elif it4 < it2 < it1:
        return swap(tour, it4, it2)  # ... it3 it4 ... it2 it1 ...
    else:
        return swap(tour, it1, it3)  # ... it2 it1 ... it3 it4 ...


@nb.njit(cache=True)
def _improve(tour: np.ndarray, matrix: np.ndarray, neighbours: np.ndarray, dlb: np.ndarray,
             it1: int, t1: int, iteration: int, set_x: set, set_y: set) -> Tuple[int, float, np.ndarray]:
    """ Последовательный 2-opt для эвристики Лина-Кернига
    tour: список городов
    matrix: матрица весов
    neighbours: набор кандидатов
    dlb: don't look bits
    it1, t1: индекс, значение города, с которого начинать
    iteration: номер итерации var-opt
    set_x, set_y: наборы удаленных, добавленных ребер
    """
    fast = len(dlb) != 1
    around_t1 = around(tour, it1)
    for it2, t2 in around_t1:  # кандидаты на t2 (их два)
        t1t2 = make_pair(t1, t2)
        candidates_t3 = neighbours[t1]

        for t3 in candidates_t3:  # кандидаты на t3 (их много)
            if t3 == t2 or (matrix[t1][t2] - matrix[t2][t3]) < 0:
                continue
            t2t3 = make_pair(t2, t3)
            if t2t3 in set_x or t2t3 in set_y:
                continue
            it3 = np.where(tour == t3)[0][0]
            around_t3 = around(tour, it3)

            for it4, t4 in around_t3:  # кандидаты на t4
                t1t4 = make_pair(t1, t4)
                if not between(tour, it1, it3, it4) or not between(tour, it4, it2, it1):
                    continue
                if t4 == t1 or t4 == t2 or t1t4 in set_x or t1t4 in set_y:
                    continue
                gain = (matrix[t1][t2] + matrix[t3][t4]) - (matrix[t2][t3] + matrix[t4][t1])
                if gain < 0:
                    continue
                tour = _get_tour(tour, it1, it2, it3, it4)  # проверяем, свапаем
                if fast:
                    dlb[t1] = dlb[t2] = dlb[t3] = dlb[t4] = False
                set_x.add(t1t2)
                set_x.add(make_pair(t3, t4))
                set_y.add(t2t3)
                set_y.add(t1t4)
                it4 = np.where(tour == t4)[0][0]
                iteration, up, tour = _improve(tour, matrix, neighbours, dlb, it4, t4, iteration + 1, set_x, set_y)
                gain += up
                return iteration, gain, tour
    return iteration, 0.0, tour


@nb.njit(cache=True)
def _double_bridge(tour: np.ndarray, matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """ Двойной мост - непоследовательный 4-opt
    tour: список городов
    matrix: матрица весов
    return: выигрыш, новый тур
    """
    best_gain, exchange, size = 0.0, None, len(tour)

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
        (x, y, z, w), idx = exchange, 0
        temp = np.zeros(size, dtype=nb.int64)
        idx = _copy_slice(temp, tour, x + 1, y, idx)
        idx = _copy_slice(temp, tour, w + 1, x, idx)
        idx = _copy_slice(temp, tour, z + 1, w, idx)
        _copy_slice(temp, tour, y + 1, z, idx)
        return best_gain, temp

    return 0., tour


@nb.njit(cache=True)
def _copy_slice(temp: np.ndarray, tour: np.ndarray, x: int, y: int, idx: int) -> int:
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


class LKOpt(AbcOpt):
    """ Локальный поиск: алгоритм Лина-Кернигана
    Вычислительная сложность поиска локального минимума: O(n^2.2)
    """

    def __init__(self, length: float, tour: np.ndarray, matrix: np.ndarray, **kwargs):
        """
        dlb: don't look bits [boolean]
        bridge: make double bridge [boolean]
        neighbours: number of neighbours [int]
        """
        super().__init__(length, tour, matrix, **kwargs)

        dlb = kwargs.get('dlb', False)
        neighbours = kwargs.get('neighbours', 5)
        self.bridge = kwargs.get('bridge', True)

        self.neighbours = self._calc_neighbours(neighbours)
        self.dlb = np.zeros(self.size if dlb else 1, dtype=bool)

    def improve(self) -> float:
        """ Локальный поиск (поиск изменения + само изменение)
        return: выигрыш от локального поиска
        """
        gain, x, y = 0.0, {(0, 0)}, {(0, 0)}

        for t1, city in enumerate(self.tour):
            if self._check_dlb(city):
                continue
            iteration, gain, tour = _improve(self.tour, self.matrix, self.neighbours, self.dlb, t1, city, 0, x, y)
            if gain > 1.e-10:
                logging.info(f'iteration k-opt : {iteration}')
                self.length -= gain
                self.tour = tour
                self.collector.update({'length': self.length, 'gain': gain})
                return gain

            if len(self.dlb) != 1:
                self.dlb[t1] = True

        if self.bridge:
            gain, tour = _double_bridge(self.tour, self.matrix)
            if gain > 1.e-10:
                logging.info(f'non-seq 4-opt')
                self.length -= gain
                self.tour = tour

                if len(self.dlb) != 1:
                    self.dlb = np.zeros(self.size, dtype=bool)

                self.collector.update({'length': self.length, 'gain': gain})
                return gain

        return 0.

    def _calc_neighbours(self, count: int) -> np.ndarray:
        """ Собираем кандидатов по приоритету соседства
        count: сколько соседей отбираем в кандидаты
        """
        assert 0 < count < self.size, 'bad count'
        neighbours = np.zeros([self.size, count], dtype=int)
        for i in self.tour:
            temp = []
            for j, dist in enumerate(self.matrix[i]):
                if dist > 0:
                    temp.append((dist, j))
            temp = sorted(temp)[:count]
            for idx, node in enumerate(temp):
                neighbours[i][idx] = node[1]
        return neighbours

    def _check_dlb(self, idx: int) -> bool:
        """ Проверка don't look bites
        idx: индекс города в туре
        """
        s = self.size
        if len(self.dlb) != 1 and self.dlb[idx] and self.dlb[(idx - 1) % s] and self.dlb[(idx + 1) % s]:
            return True
        return False
