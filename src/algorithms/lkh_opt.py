import logging
from collections import defaultdict
from typing import Tuple

import numba as nb
import numpy as np

from src.algorithms.lk_opt import __get_tour, __validation
from src.algorithms.structures.matrix import alpha_matrix
from src.algorithms.structures.one_tree import one_tree_topology
from src.algorithms.utils.abc_opt import AbcOpt
from src.algorithms.utils.double_bridge import double_bridge
from src.algorithms.utils.hash import generate_hash
from src.algorithms.utils.subgradient_optimization import SubgradientOptimization
from src.algorithms.utils.utils import around, make_pair, check_dlb


@nb.njit
def _improve(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray, dlb: np.ndarray,
             it1: int, t1: int, best: set, solutions: set, k: int) -> Tuple[float, np.ndarray]:
    """ Последовательный 2-opt для эвристики Лина-Кернига-Хельсгауна
    tour: список городов
    matrix: матрица весов
    candidates: набор кандидатов
    dlb: don't look bits
    it1, t1: индекс, значение города, с которого начинать
    solutions: полученные ранее туры
    best, set_y: наборы лушчих, добавленных ребер
    k: k-opt, k - кол-во сколько можно сделать последовательных улучшений
    return: выигрыш, новый тур
    """
    around_t1 = around(tour, it1)
    for it2, t2 in around_t1:
        t1t2 = make_pair(t1, t2)
        if t1t2 in best:
            continue

        for t3 in candidates[t2]:
            if t3 == -1:
                continue
            gain = matrix[t1][t2] - matrix[t2][t3]
            if t3 == around_t1[0][1] or t3 == around_t1[1][1] or not gain > 1.e-10:
                continue

            it3 = np.where(tour == t3)[0][0]
            set_y = {make_pair(t2, t3)}
            _gain, _tour = __choose_t4(tour, matrix, it1, it2, it3, candidates, gain, set_y, dlb, solutions, k)
            if _gain > 1.e-10:
                return _gain, _tour

    return 0., tour


@nb.njit
def __choose_t4(tour: np.ndarray, matrix: np.ndarray, it1: int, it2: int, it3: int, candidates: np.ndarray,
                gain: float, set_y: set, dlb: np.ndarray, sol: set, k: int) -> Tuple[float, np.ndarray]:
    """ Выбираем город t2i - город, который создаст ребро на удаление
    tour: список городов
    matrix: матрица весов
    it1, it2, it3: города t1, t2i, t2i+1, их индексы
    candidates: набор кандидатов
    gain: текущий выигрыш
    set_y: набор добавленных ребер
    dlb: don't look bits
    sol: существующие решения
    k: k-opt, k - кол-во сколько можно сделать последовательных улучшений
    return: выигрыш, новый тур
    """
    t1, t2, t3 = tour[it1], tour[it2], tour[it3]
    around_t3 = around(tour, it3)

    for it4, t4 in around_t3:
        t3t4 = make_pair(t3, t4)
        if t3t4 in set_y:
            continue
        if not __validation(len(tour), it1, it2, it3, it4):  # проверяем на корректность
            continue

        _tour = tour.copy()
        _it1, _it4 = __get_tour(_tour, it1, it2, it3, it4)  # единственное место, где меняется тур

        _set_y = set_y.copy()
        _set_y.add(make_pair(t1, t4))

        if generate_hash(_tour) in sol:  # проверяем, был ли такой раньше
            continue

        _gain = gain + (matrix[t3][t4] - matrix[t1][t4])
        if _gain > 1.e-10:
            if len(dlb) != 1:
                dlb[t1] = dlb[t2] = dlb[t3] = dlb[t4] = False
            return _gain, _tour
        elif len(_set_y) <= k:
            _gain, _tour = __choose_t5(_tour, matrix, _it1, _it4, candidates, _gain, _set_y, dlb, sol, k)
            if _gain > 1.e-10:
                if len(dlb) != 1:
                    dlb[t1] = dlb[t2] = dlb[t3] = dlb[t4] = False
                return _gain, _tour
        else:
            break

    return 0., tour


@nb.njit
def __choose_t5(tour: np.ndarray, matrix: np.ndarray, it1: int, it4: int, candidates: np.ndarray,
                gain: float, set_y: set, dlb: np.ndarray, sol: set, k: int) -> Tuple[float, np.ndarray]:
    """ Выбираем город t2i+1 - город, который создаст ребро на добавление
    tour: список городов
    matrix: матрица весов
    it1, it4: города t1 и t2i, их индексы
    candidates: набор кандидатов
    gain: текущий выигрыш
    set_y: набор добавленных ребер
    dlb: don't look bits
    sol: существующие решения
    k: k-opt, k - кол-во сколько можно сделать последовательных улучшений
    return: выигрыш, новый тур
    """
    t1, t4 = tour[it1], tour[it4]
    around_t1 = around(tour, t1)
    for t5 in candidates[t4]:
        if t5 == -1 or t5 == around_t1[0][1] or t5 == around_t1[1][1]:
            continue

        t4t5 = make_pair(t4, t5)

        _gain = gain + (matrix[t1][t4] - matrix[t4][t5])
        if not _gain > 1.e-10:
            continue

        _set_y = set_y.copy()
        _set_y.add(t4t5)

        it5 = np.where(tour == t5)[0][0]
        _gain, _tour = __choose_t4(tour, matrix, it1, it4, it5, candidates, _gain, _set_y, dlb, sol, k)
        if _gain > 1.e-10:
            return _gain, _tour

    return 0., tour


class LKHOpt(AbcOpt):
    """ Локальный поиск: алгоритм Лина-Кернигана
    Вычислительная сложность поиска локального минимума: O(n^2.6)?
    Обладает улучшенной эвристикой поиска кандидатов

    length: начальная длина тура
    tour: начальный тур
    matrix: матрица весов

    dlb: don't look bits [boolean]
    bridge: make double bridge [tuple]
    excess: parameter for cut bad candidates [float]
    mul: excess factor
    k: number of k for k-opt; how many sequential can make algorithm [int]
    subgradient: use or not subgradient optimization
    """

    def __init__(self, length: float, tour: np.ndarray, matrix: np.ndarray, **kwargs):
        super().__init__(length, tour, matrix, **kwargs)

        subgradient = kwargs.get('subgradient', False)
        if subgradient:
            self.gradient = SubgradientOptimization.run(self.matrix)
            SubgradientOptimization.make_move(self.gradient.pi_sum, self.matrix)
            logging.info('subgradient optimization done')
            _length, _f, _s, self.best_solution, topology = one_tree_topology(self.matrix)
            self.alpha = alpha_matrix(self.matrix, _f, _s, topology)
            logging.info('alpha-matrix done')
            SubgradientOptimization.get_back(self.gradient.pi_sum, self.matrix)
        else:
            _length, _f, _s, self.best_solution, topology = one_tree_topology(self.matrix)
            self.alpha = alpha_matrix(self.matrix, _f, _s, topology)
            logging.info('alpha-matrix done')

        dlb = kwargs.get('dlb', True)
        self.k = kwargs.get('k', 5)
        self.excess = kwargs.get('mul', 1) * kwargs.get('excess', 1 / self.size * _length)
        self.bridge = kwargs.get('bridge', True)

        self.candidates = self._calc_candidates(self.tour, self.alpha, self.matrix, self.excess)
        self.dlb = np.zeros(self.size if dlb else 1, dtype=bool)
        logging.info('initialization lkh done')

    @staticmethod
    def _calc_candidates(tour: np.ndarray, alpha: np.ndarray, matrix: np.ndarray, excess: float) -> np.ndarray:
        """ Отбираем кандидатов по альфа-мере
        tour: список городов
        alpha: матрица альфа-мер
        matrix: матрица весов
        excess: уровень по которому отбираем кандидатов
        return: матрица кандидатов [size * max(num of candidates for i)], пустое заполнено -1
        """
        max_num, size, candidates = 0, len(matrix), defaultdict(list)
        for i in tour:
            for j, dist in enumerate(matrix[i]):
                if alpha[i][j] < excess:
                    candidates[i].append((alpha[i][j], matrix[i][j], j))
            if max_num < len(candidates[i]):
                max_num = len(candidates[i])
            candidates[i].sort()
        temp = np.full((size, max_num), -1, dtype=np.int64)
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
        for it1, t1 in enumerate(self.tour):
            if check_dlb(self.dlb, t1):
                continue
            gain, tour = _improve(
                self.tour, self.matrix, self.candidates, self.dlb, it1, t1, self.best_solution, self.solutions, self.k
            )
            if gain > 1.e-10:
                logging.info('iteration k-opt')
                self.tour = tour
                self.length -= gain
                if self.collector is not None:
                    self.collector.update({'length': self.length, 'gain': gain})
                return gain

            if len(self.dlb) != 1:
                self.dlb[t1] = True

        if self.bridge:
            gain, tour = double_bridge(self.tour, self.matrix, self.candidates, True)

            if gain > 1.e-10:
                logging.info('non-seq 4-opt')
                self.tour = tour
                self.length -= gain

                if len(self.dlb) != 1:
                    self.dlb = np.zeros(self.size, dtype=bool)

                if self.collector is not None:
                    self.collector.update({'length': self.length, 'gain': gain})
                return gain

        return 0.
