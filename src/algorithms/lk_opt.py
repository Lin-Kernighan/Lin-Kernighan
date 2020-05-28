import logging
import warnings
from typing import Tuple

import numba as nb
import numpy as np
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from src.algorithms.utils.abc_opt import AbcOpt
from src.algorithms.utils.double_bridge import double_bridge
from src.algorithms.utils.hash import generate_hash
from src.algorithms.utils.utils import swap, around, make_pair, between, check_dlb

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@nb.njit(cache=True)
def __get_tour(tour: np.ndarray, it1: int, it2: int, it3: int, it4: int) -> tuple:
    """ Выполняем k-opt для указанных города
    tour: список городов
    it1, it2, it3, it4: найденные города
    return: it1, it4
    """
    if it2 < it4 < it3 and (it1 < it2 or it3 < it1):
        swap(tour, it2, it4)  # ... it1 it2 ... it4 it3 ... && it2 ... it4 it3 ... it1
        return it1, it2
    elif it3 < it1 < it2 and (it4 < it3 or it2 < it4):
        swap(tour, it3, it1)  # ... it4 it3 ... it1 it2 ... && it3 ... it1 it2 ... it4
        return it3, it4
    elif it4 < it2 < it1 and (it3 < it4 or it1 < it3):
        swap(tour, it4, it2)  # ... it3 it4 ... it2 it1 ... && it4 ... it2 it1 ... it3
        return it1, it2
    elif it1 < it3 < it3 and (it2 < it1 or it4 < it2):
        swap(tour, it1, it3)  # ... it2 it1 ... it3 it4 ... && it1 ... it3 it4 ... it2
        return it3, it4
    else:
        assert False, 'bad tour'


@nb.njit(cache=True)
def __validation(size: int, it1: int, it2: int, it3: int, it4: int) -> bool:
    """ Проверка на корректность тура
    size: размер маршрута
    it1, it2, it3, it4: индексы городов: t1, t2i, t2i+1, t2i+2
    return: корректен или нет
    """
    return between(size, it1, it3, it4) and between(size, it4, it2, it1)


@nb.njit
def _improve(tour: np.ndarray, matrix: np.ndarray, neighbours: np.ndarray, dlb: np.ndarray,
             it1: int, t1: int, solutions: set, k: int) -> Tuple[float, np.ndarray]:
    """ Последовательный 2-opt для эвристики Лина-Кернига
    tour: список городов
    matrix: матрица весов
    neighbours: набор кандидатов
    dlb: don't look bits
    it1, t1: индекс, значение города, с которого начинать
    solutions: полученные ранее туры
    set_x, set_y: наборы удаленных, добавленных ребер
    k: k-opt, k - кол-во сколько можно сделать последовательных улучшений
    return: выигрыш, новый тур
    """
    around_t1 = around(tour, it1)
    for it2, t2 in around_t1:
        set_x = {make_pair(t1, t2)}

        for t3 in neighbours[t2]:
            gain = matrix[t1][t2] - matrix[t2][t3]
            if t3 == around_t1[0][1] or t3 == around_t1[1][1] or not gain > 1.e-10:
                continue
            set_y = {make_pair(t2, t3)}
            it3 = np.where(tour == t3)[0][0]
            _gain, _tour = __choose_t4(tour, matrix, it1, it2, it3, neighbours, gain, set_x, set_y, dlb, solutions, k)
            if _gain > 1.e-10:
                return _gain, _tour

    return 0., tour


@nb.njit
def __choose_t4(tour: np.ndarray, matrix: np.ndarray, it1: int, it2: int, it3: int, neighbours: np.ndarray,
                gain: float, set_x: set, set_y: set, dlb: np.ndarray, sol: set, k: int) -> Tuple[float, np.ndarray]:
    """ Выбираем город t2i - город, который создаст ребро на удаление
    tour: список городов
    matrix: матрица весов
    it1, it2, it3: города t1, t2i, t2i+1, их индексы
    neighbours: набор кандидатов
    gain: текущий выигрыш
    set_x, set_y: наборы удаленных, добавленных ребер
    dlb: don't look bits
    sol: существующие решения
    k: k-opt, k - кол-во сколько можно сделать последовательных улучшений
    return: выигрыш, новый тур
    """
    t1, t2, t3 = tour[it1], tour[it2], tour[it3]
    around_t3 = around(tour, it3)

    for it4, t4 in around_t3:
        if len(set_y) == k - 1:  # выбираем длиннейшее ребро на последней итерации
            if matrix[t3][around_t3[0][1]] < matrix[t3][around_t3[1][1]] and around_t3[0][1] == t4:
                break
            if matrix[t3][around_t3[0][1]] > matrix[t3][around_t3[1][1]] and around_t3[0][1] == t4:
                break

        t3t4 = make_pair(t3, t4)
        if t3t4 in set_x or t3t4 in set_y:
            continue
        if not __validation(len(tour), it1, it2, it3, it4):  # проверяем на корректность
            continue

        _set_x = set_x.copy()
        _set_y = set_y.copy()
        _set_x.add(t3t4)
        _set_y.add(make_pair(t1, t4))

        _tour = tour.copy()
        _it1, _it4 = __get_tour(_tour, it1, it2, it3, it4)  # единственное место, где меняется тур

        if generate_hash(_tour) in sol:  # проверяем, был ли такой раньше
            continue

        _gain = gain + (matrix[t3][t4] - matrix[t1][t4])
        if _gain > 1.e-10:
            if len(dlb) != 1:
                dlb[t1] = dlb[t2] = dlb[t3] = dlb[t4] = False
            return _gain, _tour
        elif len(_set_x) <= k:
            _gain, _tour = __choose_t5(_tour, matrix, _it1, _it4, neighbours, _gain, _set_x, _set_y, dlb, sol, k)
            if _gain > 1.e-10:
                if len(dlb) != 1:
                    dlb[t1] = dlb[t2] = dlb[t3] = dlb[t4] = False
                return _gain, _tour
        else:
            break

    return 0., tour


@nb.njit
def __choose_t5(tour: np.ndarray, matrix: np.ndarray, it1: int, it4: int, neighbours: np.ndarray,
                gain: float, set_x: set, set_y: set, dlb: np.ndarray, sol: set, k: int) -> Tuple[float, np.ndarray]:
    """ Выбираем город t2i+1 - город, который создаст ребро на добавление
    tour: список городов
    matrix: матрица весов
    it1, it4: города t1 и t2i, их индексы
    neighbours: набор кандидатов
    gain: текущий выигрыш
    set_x, set_y: наборы удаленных, добавленных ребер
    dlb: don't look bits
    sol: существующие решения
    k: k-opt, k - кол-во сколько можно сделать последовательных улучшений
    return: выигрыш, новый тур
    """
    t1, t4 = tour[it1], tour[it4]
    around_t1 = around(tour, t1)
    for t5 in neighbours[t4]:
        if t5 == around_t1[0][1] or t5 == around_t1[1][1]:
            continue

        t4t5 = make_pair(t4, t5)

        _gain = gain + (matrix[t1][t4] - matrix[t4][t5])
        if not _gain > 1.e-10 or t4t5 in set_x or t4t5 in set_y:
            continue

        _set_y = set_y.copy()
        _set_y.add(t4t5)

        it5 = np.where(tour == t5)[0][0]
        _gain, _tour = __choose_t4(tour, matrix, it1, it4, it5, neighbours, _gain, set_x, _set_y, dlb, sol, k)
        if _gain > 1.e-10:
            return _gain, _tour

    return 0., tour


class LKOpt(AbcOpt):
    """ Локальный поиск: алгоритм Лина-Кернигана
    Вычислительная сложность поиска локального минимума: O(n^2.2)

    length: начальная длина тура
    tour: начальный тур
    matrix: матрица весов

    dlb: don't look bits [boolean]
    bridge: make double bridge [tuple] ([not use: 0, all cities: 1, only neighbours: 2], fast scheme)
    neighbours: number of neighbours [int]
    k: number of k for k-opt; how many sequential can make algorithm [int]
    """

    def __init__(self, length: float, tour: np.ndarray, matrix: np.ndarray, **kwargs):
        super().__init__(length, tour, matrix, **kwargs)

        dlb = kwargs.get('dlb', False)
        neighbours = kwargs.get('neighbours', 5)
        self.k = kwargs.get('k', 5)
        self.bridge, self.fast = kwargs.get('bridge', (2, True))

        self.neighbours = self._calc_neighbours(neighbours)
        self.dlb = np.zeros(self.size if dlb else 1, dtype=bool)

    def improve(self) -> float:
        """ Локальный поиск (поиск изменения + само изменение)
        return: выигрыш от локального поиска
        """
        for it1, t1 in enumerate(self.tour):
            if check_dlb(self.dlb, t1):
                continue
            gain, tour = _improve(self.tour, self.matrix, self.neighbours, self.dlb, it1, t1, self.solutions, self.k)
            if gain > 1.e-10:
                logging.info('iteration k-opt')
                self.length -= gain
                self.tour = tour
                if self.collector is not None:
                    self.collector.update({'length': self.length, 'gain': gain})
                return gain

            if len(self.dlb) != 1:
                self.dlb[t1] = True

        if self.bridge != 0:
            gain, tour = 0, None

            if self.bridge == 1:
                gain, tour = double_bridge(self.tour, self.matrix, np.zeros([2, 2], dtype=int), self.fast)
            elif self.bridge == 2:
                gain, tour = double_bridge(self.tour, self.matrix, self.neighbours, self.fast)

            if gain > 1.e-10:
                logging.info('non-seq 4-opt')
                self.length -= gain
                self.tour = tour

                if len(self.dlb) != 1:
                    self.dlb = np.zeros(self.size, dtype=bool)

                if self.collector is not None:
                    self.collector.update({'length': self.length, 'gain': gain})
                return gain

        return 0.

    def _calc_neighbours(self, count: int) -> np.ndarray:
        """ Собираем кандидатов по приоритету соседства
        count: сколько соседей отбираем в кандидаты
        return: матрица кандидатов
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
