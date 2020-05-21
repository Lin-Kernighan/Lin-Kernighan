import logging
from collections import Set

import numba as nb
import numpy as np

from src.algorithms.heuristics.abc_opt import AbcOpt
from src.utils import make_pair, get_hash, swap, between, around


@nb.njit
def _check_tour(tour: np.ndarray, it1: int, it2: int, it3: int, it4: int) -> np.ndarray:
    tour = tour.copy()

    if not between(tour, it1, it3, it4) or not between(tour, it4, it2, it1):
        return np.zeros(1, dtype=nb.int64)
    if it1 < it2 < it4:
        return swap(tour, it2, it4)  # ... it1 it2 ... it4 it3 ...
    elif it4 < it3 < it1:
        return swap(tour, it3, it1)  # ... it4 it3 ... it1 it2 ...
    elif it4 < it2 < it1:
        return swap(tour, it4, it2)  # ... it3 it4 ... it2 it1 ...
    else:
        return swap(tour, it1, it3)  # ... it2 it1 ... it3 it4 ...


class LKOpt(AbcOpt):
    """ Локальный поиск: алгоритм Лина-Кернигана
    Вычислительная сложность поиска локального минимума: O(n^2.2)
    """

    def __init__(self, length: float, tour: np.ndarray, matrix: np.ndarray, **kwargs):
        super().__init__(length, tour, matrix, **kwargs)
        self.solutions: Set[int] = {get_hash(self.tour)}

        dlb = kwargs.get('dlb', False)
        neighbours = kwargs.get('radius', 5)

        self.neighbours = self._calc_neighbours(neighbours)
        self.dlb = np.zeros(self.size if dlb else 1, dtype=bool)

    def improve(self) -> float:
        gain, set_x, set_y = 0.0, set(), set()

        for t1, city in enumerate(self.tour):
            if self._check_dlb(city):
                continue
            gain = self._improve(t1, city, 0, set_x, set_y)
            if gain > 0.0:
                self.length -= gain
                return gain
            if len(self.dlb) != 1:
                self.dlb[t1] = True

        return gain

    def _calc_neighbours(self, count: int) -> np.ndarray:
        """ Собираем кандидатов по приоритету соседства """
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

    def _check_dlb(self, t: int) -> bool:
        s = self.size
        if len(self.dlb) != 1 and self.dlb[t] and self.dlb[(t - 1) % s] and self.dlb[(t + 1) % s]:
            return True
        return False

    def _improve(self, it1: int, t1: int, iteration: int, set_x: set, set_y: set) -> float:
        fast = len(self.dlb) != 1
        around_t1 = around(self.tour, it1)
        for it2, t2 in around_t1:  # кандидаты на t2 (их два)
            t1t2 = make_pair(t1, t2)
            candidates_t3 = self.neighbours[t1]

            for t3 in candidates_t3:  # кандидаты на t3 (их много)
                if t3 == t2 or (self.matrix[t1][t2] - self.matrix[t2][t3]) < 0:
                    continue
                t2t3 = make_pair(t2, t3)
                if t2t3 in set_x or t2t3 in set_y:
                    continue
                it3 = np.where(self.tour == t3)[0][0]
                around_t3 = around(self.tour, it3)

                for it4, t4 in around_t3:  # кандидаты на t4 (их два - подходит один, оптимизация?)
                    t1t4 = make_pair(t1, t4)
                    if t4 == t1 or t4 == t2 or t1t4 in set_x or t1t4 in set_y:
                        continue
                    gain = (self.matrix[t1][t2] + self.matrix[t3][t4]) - (self.matrix[t2][t3] + self.matrix[t4][t1])
                    if gain < 0:
                        continue
                    temp = _check_tour(self.tour, it1, it2, it3, it4)  # проверяем, свапаем
                    if len(temp) == 1:
                        continue

                    self.tour = temp.copy()  # иначе все хорошо, присваиваем
                    if fast:
                        self.dlb[t1] = self.dlb[t2] = self.dlb[t3] = self.dlb[t4] = False
                    set_x.add(t1t2)
                    set_x.add(make_pair(t3, t4))
                    set_y.add(t2t3)
                    set_y.add(t1t4)
                    it4 = np.where(self.tour == t4)[0][0]
                    logging.info(f'iteration k-opt : {iteration}')
                    gain += self._improve(it4, t4, iteration + 1, set_x, set_y)
                    return gain
        return 0.0
