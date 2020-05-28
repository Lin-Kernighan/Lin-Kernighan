from typing import Tuple

import numba as nb
import numpy as np

from src.algorithms.utils.utils import between, around


@nb.njit(cache=True)
def non_sequential_move(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray) -> Tuple[float, np.ndarray]:
    _gain, _tour = __ns_two_opt(tour, matrix, candidates)
    if _gain > 1.e-10:
        return _gain, _tour
    _gain, _tour = __ns_three_opt(tour, matrix, candidates)
    if _gain > 1.e-10:
        return _gain, _tour
    return 0., tour


@nb.njit(cache=True)
def __ns_two_opt(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray) -> Tuple[float, np.ndarray]:
    size = len(tour)
    towns = np.full((2, 10), -1, dtype=nb.int64)  # [[t1, t2, t3, t4 ... t10], [it1, it2 ... it10]]
    # towns собирает кандидатов по ходу выполнения, так как операции рекурсивные, в массиве может оказаться мусор

    for it1 in range(size):
        it2 = (it1 + 1) % size
        t1, t2 = tour[it1], tour[it2]
        for t3 in candidates[t2]:
            if t3 == -1 or t3 == t1 or t3 == t2 \
                    or tour[(it1 - 1) % size] == t3 or tour[(it1 - 2) % size] == t3 or tour[(it1 - 3) % size] == t3 \
                    or tour[(it2 + 1) % size] == t3 or tour[(it2 + 2) % size] == t3:  # cheaper to check
                continue
            it3 = np.where(tour == t3)[0][0]
            it4 = (it3 + 1) % size
            t4 = tour[it4]
            towns[0][0], towns[0][1], towns[0][2], towns[0][3] = t1, t2, t3, t4
            towns[1][0], towns[1][1], towns[1][2], towns[1][3] = it1, it2, it3, it4
            _gain, _tour = __choose_t5(tour, matrix, candidates, towns)
            if _gain > 1.e-10:
                return _gain, _tour

    return 0., tour


@nb.njit(cache=True)
def __choose_t5(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray, towns: np.ndarray) -> tuple:
    t1, t2, t3, t4 = towns[0][0], towns[0][1], towns[0][2], towns[0][3]
    it1, it2, it3, it4 = towns[1][0], towns[1][1], towns[1][2], towns[1][3]

    for t5 in candidates[t4]:
        if t5 in (-1, t1, t2, t3):
            continue
        it5 = np.where(tour == t5)[0][0]
        flag = False if between(len(tour), it2, it3, it5) else True
        for it6, t6 in around(tour, it5):
            if t6 in (t1, t2, t3, t4):
                continue
            towns[0][4], towns[0][5] = t5, t6
            towns[1][4], towns[1][5] = it5, it6
            _gain, _tour = __choose_t7(tour, matrix, candidates, towns, flag)
            if _gain > 1.e-10:
                return _gain, _tour

    return 0., tour


@nb.njit(cache=True)
def __choose_t7(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray, towns: np.ndarray, flag: bool) -> tuple:
    t1, t2, t3, t4, t5, t6 = towns[0][0], towns[0][1], towns[0][2], towns[0][3], towns[0][4], towns[0][5]
    it1, it2, it3, it4, it5, it6 = towns[1][0], towns[1][1], towns[1][2], towns[1][3], towns[1][4], towns[1][5]

    for t7 in candidates[t6]:
        if t5 in (-1, t1, t2, t3, t4, t5):
            continue
        it7 = np.where(tour == t7)[0][0]
        _flag = flag if flag is True else False if between(len(tour), it2, it3, it7) else True
        for it8, t8 in around(tour, it7):
            if t8 in (t1, t2, t3, t4, t5, t6):
                continue
            towns[0][6], towns[0][7] = t7, t8
            towns[1][6], towns[1][7] = it7, it8
            _gain, _tour = __choose_t9(tour, matrix, candidates, towns, flag)
            if _gain > 1.e-10:
                return _gain, _tour

    return 0., tour


@nb.njit(cache=True)
def __choose_t9(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray, towns: np.ndarray, flag: bool) -> tuple:
    t1, t2, t3, t4, t5, t6, t7, t8 =\
        towns[0][0], towns[0][1], towns[0][2], towns[0][3], towns[0][4], towns[0][5], towns[0][6], towns[0][7]
    it1, it2, it3, it4, it5, it6, it7, it8 =\
        towns[1][0], towns[1][1], towns[1][2], towns[1][3], towns[1][4], towns[1][5], towns[1][6], towns[1][7]

    for t9 in candidates[t8]:
        if t9 in (-1, t1, t2, t3, t4, t5, t6, t7):
            continue
        it9 = np.where(tour == t9)[0][0]
        _flag = flag if flag is True else False if between(len(tour), it2, it3, it9) else True
        if _flag is False:
            continue
        for it10, t10 in around(tour, it9):
            if t10 in (t1, t2, t3, t4, t5, t6, t7, t8, t9):
                continue
            towns[0][8], towns[0][9] = t9, t10
            towns[1][8], towns[1][9] = it7, it8
            _gain = __get_gain(matrix, towns)
            if _gain > 1.e-10:
                return _gain, None

    return 0., tour


@nb.njit(cache=True)
def __get_gain(matrix: np.ndarray, towns: np.ndarray) -> float:
    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 =\
        towns[0][0], towns[0][1], towns[0][2], towns[0][3], towns[0][4],\
        towns[0][5], towns[0][6], towns[0][7], towns[0][8], towns[0][9]
    gain = matrix[t1][t2] + matrix[t3][t4] + matrix[t5][t6] + matrix[t7][t8] + matrix[t9][t10]
    gain -= (matrix[t4][t5] + matrix[t6][t7] + matrix[t8][t9] + matrix[t1][t10] + matrix[t2][t3])
    # проблема собрать обратно
    return gain


@nb.njit(cache=True)
def __ns_three_opt(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray) -> Tuple[float, np.ndarray]:
    return 0., tour
