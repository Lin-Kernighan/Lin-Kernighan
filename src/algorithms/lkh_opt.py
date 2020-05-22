import logging
from collections import defaultdict
from typing import Tuple

import numba as nb
import numpy as np

from src.algorithms.lk_opt import __get_tour
from src.algorithms.structures.matrix import alpha_matrix
from src.algorithms.utils.abc_opt import AbcOpt
from src.algorithms.utils.double_bridge import double_bridge
from src.algorithms.utils.subgradient_optimization import SubgradientOptimization
from src.algorithms.structures.one_tree import one_tree_topology
from src.algorithms.utils.utils import around, make_pair, between, check_dlb


@nb.njit(cache=True)
def _improve(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray, dlb: np.ndarray,
             it1: int, t1: int, iteration: int, best: set, set_y: set) -> Tuple[int, float, np.ndarray]:
    """ Последовательный 2-opt для эвристики Лина-Кернига-Хельсгауна
    tour: список городов
    matrix: матрица весов
    candidates: набор кандидатов
    dlb: don't look bits
    it1, t1: индекс, значение города, с которого начинать
    iteration: номер итерации var-opt
    best, set_y: наборы лучших, добавленных ребер
    """
    fast = len(dlb) != 1
    around_t1 = around(tour, it1)
    for it2, t2 in around_t1:  # кандидаты на t2 (их два)
        t1t2 = make_pair(t1, t2)
        if iteration == 0 and t1t2 in best:
            continue
        candidates_t3 = candidates[t1]

        for t3 in candidates_t3:  # кандидаты на t3 (их много)
            if t3 == t2 or t3 == -1 or (matrix[t1][t2] - matrix[t2][t3]) < 0:
                continue
            t2t3 = make_pair(t2, t3)
            it3 = np.where(tour == t3)[0][0]
            around_t3 = around(tour, it3)

            for it4, t4 in around_t3:  # кандидаты на t4
                t1t4 = make_pair(t1, t4)
                if t4 == t1 or t4 == t2:
                    continue
                if t1t4 in set_y:
                    continue
                if not between(tour, it1, it3, it4) or not between(tour, it4, it2, it1):
                    continue
                gain = (matrix[t1][t2] + matrix[t3][t4]) - (matrix[t2][t3] + matrix[t4][t1])
                if gain <= 0:
                    continue
                tour = __get_tour(tour, it1, it2, it3, it4)  # проверяем, свапаем
                if fast:
                    dlb[t1] = dlb[t2] = False
                    dlb[t3] = dlb[t4] = False
                it4 = np.where(tour == t4)[0][0]
                set_y.add(t1t4)
                set_y.add(t2t3)
                iteration, up, tour = _improve(tour, matrix, candidates, dlb, it4, t4, iteration + 1, best, set_y)
                gain += up
                return iteration, gain, tour
    return iteration, 0.0, tour


class LKHOpt(AbcOpt):

    def __init__(self, length: float, tour: np.ndarray, matrix: np.ndarray, **kwargs):
        super().__init__(length, tour, matrix, **kwargs)

        self.gradient = SubgradientOptimization.run(self.matrix)
        SubgradientOptimization.make_move(self.gradient.pi_sum, self.matrix)
        _length, _f, _s, self.best_solution, topology = one_tree_topology(self.matrix)
        self.alpha = alpha_matrix(self.matrix, _f, _s, topology)
        SubgradientOptimization.get_back(self.gradient.pi_sum, self.matrix)

        dlb = kwargs.get('dlb', False)
        excess = kwargs.get('excess', 1 / self.size * _length)
        self.bridge, self.fast = kwargs.get('bridge', (2, True))

        self.candidates = defaultdict(list)
        self.candidates = self._calc_candidates(self.tour, self.alpha, self.matrix, excess)
        self.dlb = np.zeros(self.size if dlb else 1, dtype=bool)

    @staticmethod
    def _calc_candidates(tour: np.ndarray, alpha: np.ndarray, matrix: np.ndarray, excess: float) -> np.ndarray:
        max_num, size, candidates = 0, len(matrix), defaultdict(list)
        for i in tour:
            for j, dist in enumerate(matrix[i]):
                if alpha[i][j] < excess:
                    candidates[i].append((alpha[i][j], matrix[i][j], j))
            if max_num < len(candidates[i]):
                max_num = len(candidates[i])
            candidates[i].sort()
        temp = np.full((size, max_num), -1, dtype=int)
        for i, candidate in candidates.items():
            idx = 0
            for _, _, j in candidate:
                temp[i][idx] = j
                idx += 1
        return temp

    def improve(self) -> float:
        """ Локальный поиск (поиск изменения + само изменение)
        return: выигрыш от локального поиска
        """
        gain, y = 0.0, {(0, 0)}

        for t1, city in enumerate(self.tour):
            if check_dlb(self.dlb, city):
                continue
            iteration, gain, tour = _improve(
                self.tour, self.matrix, self.candidates, self.dlb, t1, city, 0, self.best_solution, y
            )
            if gain > 1.e-10:
                logging.info(f'iteration k-opt : {iteration}')
                self.tour = tour
                self.length -= gain
                self.collector.update({'length': self.length, 'gain': gain})
                return gain

            if len(self.dlb) != 1:
                self.dlb[t1] = True

        if self.bridge != 0:
            gain, tour = double_bridge(self.tour, self.matrix, self.candidates, self.fast)

            if gain > 1.e-10:
                logging.info(f'non-seq 4-opt')
                self.length -= gain
                self.tour = tour

                if len(self.dlb) != 1:
                    self.dlb = np.zeros(self.size, dtype=bool)

                self.collector.update({'length': self.length, 'gain': gain})
                return gain

        return 0.

    # def lkh_optimize(self, iterations=10) -> np.ndarray:
    #     self.optimize()
    #     best_length, best_tour = self.length, self.tour
    #     best_solution = get_set(self.tour)
    #
    #     for _ in range(iterations):
    #         self.dlb = np.zeros(self.size, dtype=bool) if self.dlb is not None else None
    #         self.length, self.tour = \
    #             InitialTour.helsgaun(self.alpha, self.matrix, best_solution, self.candidates, self.excess)
    #         self.temp_length = self.length
    #         self.optimize()
    #         if self.length < best_length:
    #             best_length, best_tour = self.length, self.tour
    #             best_solution = get_set(self.tour)
    #
    #     self.length, self.tour = best_length, best_tour
    #     return self.tour
