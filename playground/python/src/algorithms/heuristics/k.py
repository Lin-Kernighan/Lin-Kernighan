import logging
from collections import Set
from typing import Tuple

import numba as nb
import numpy as np

from src.algorithms.heuristics.abc_opt import AbcOpt
from src.utils import make_pair, get_hash, swap, between, around


@nb.njit
def _get_tour(tour: np.ndarray, it1: int, it2: int, it3: int, it4: int) -> np.ndarray:
    if it1 < it2 < it4:
        return swap(tour, it2, it4)  # ... it1 it2 ... it4 it3 ...
    elif it4 < it3 < it1:
        return swap(tour, it3, it1)  # ... it4 it3 ... it1 it2 ...
    elif it4 < it2 < it1:
        return swap(tour, it4, it2)  # ... it3 it4 ... it2 it1 ...
    else:
        return swap(tour, it1, it3)  # ... it2 it1 ... it3 it4 ...


@nb.njit
def _improve(tour: np.ndarray, matrix: np.ndarray, neighbours: np.ndarray, dlb: np.ndarray,
             it1: int, t1: int, iteration: int, set_x: set, set_y: set) -> Tuple[int, float, np.ndarray]:
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
        gain, x, y = 0.0, {(0, 0)}, {(0, 0)}

        for t1, city in enumerate(self.tour):
            if self._check_dlb(city):
                continue
            iteration, gain, tour = _improve(self.tour, self.matrix, self.neighbours, self.dlb, t1, city, 0, x, y)
            if gain > 0.0:
                logging.info(f'iteration k-opt : {iteration}')
                self.length -= gain
                self.tour = tour.copy()
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
