from typing import Tuple

import numba as nb
import numpy as np

from src.algorithms.utils.utils import between, around, make_pair


@nb.njit(cache=True)
def non_sequential_move(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray) -> Tuple[float, np.ndarray]:
    _gain, _tour = __ns_two_opt(tour, matrix, candidates)
    if _gain > 1.e-10:
        return _gain, _tour
    return 0., tour


@nb.njit(cache=True)
def __ns_two_opt(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray) -> Tuple[float, np.ndarray]:
    """ Разбиение тура начать с 2-opt """
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
    """ Выбираем t5-t6 ребро на удаление, t4-t5 кандидат """
    t1, t2, t3, t4 = towns[0][0], towns[0][1], towns[0][2], towns[0][3]
    it2, it3, size = towns[1][1], towns[1][2], len(tour)

    for t5 in candidates[t4]:
        if t5 in (-1, t1, t2, t3):
            continue
        it5 = np.where(tour == t5)[0][0]
        flag = False if between(size, it2, it3, it5) else True
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
    """ Выбираем t7-t8 ребро на удаление, t6-t7 кандидат """
    t1, t2, t3, t4, t5, t6 = towns[0][0], towns[0][1], towns[0][2], towns[0][3], towns[0][4], towns[0][5]
    it2, it3, size = towns[1][1], towns[1][2], len(tour)

    for t7 in candidates[t6]:
        if t7 in (-1, t1, t2, t3, t4, t5):
            continue
        it7 = np.where(tour == t7)[0][0]
        _flag = flag if flag is True else False if between(size, it2, it3, it7) else True
        for it8, t8 in around(tour, it7):
            if t8 in (t1, t2, t3, t4, t5, t6):
                continue
            towns[0][6], towns[0][7] = t7, t8
            towns[1][6], towns[1][7] = it7, it8
            _gain, _tour = __choose_t9(tour, matrix, candidates, towns, _flag)
            if _gain > 1.e-10:
                return _gain, _tour

    return 0., tour


@nb.njit(cache=True)
def __choose_t9(tour: np.ndarray, matrix: np.ndarray, candidates: np.ndarray, towns: np.ndarray, flag: bool) -> tuple:
    """ Выбираем t9-t10 ребро на удаление, t8-t9 кандидат """
    t1, t2, t3, t4, t5, t6, t7, t8 = \
        towns[0][0], towns[0][1], towns[0][2], towns[0][3], towns[0][4], towns[0][5], towns[0][6], towns[0][7]
    it2, it3, size = towns[1][1], towns[1][2], len(tour)

    for t9 in candidates[t8]:
        if t9 in (-1, t1, t2, t3, t4, t5, t6, t7):
            continue
        it9 = np.where(tour == t9)[0][0]
        _flag = flag if flag is True else False if between(size, it2, it3, it9) else True
        if _flag is False:
            continue
        for it10, t10 in around(tour, it9):
            if t10 in (t1, t2, t3, t4, t5, t6, t7, t8, t9):
                continue
            towns[0][8], towns[0][9] = t9, t10
            towns[1][8], towns[1][9] = it9, it10
            _gain = __get_gain(matrix, towns)
            if _gain > 1.e-10:
                is_tour, _tour = __get_tour(tour, towns)
                if is_tour:
                    return _gain, _tour

    return 0., tour


@nb.njit(cache=True)
def __get_gain(matrix: np.ndarray, towns: np.ndarray) -> float:
    """ Удаляем ребра - добавляем ребра
    matrix: матрица весов
    towns: собранные города в правильной последовательности
    return: выигрыш теоретический
    """
    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = towns[0]
    gain = matrix[t1][t2] + matrix[t3][t4] + matrix[t5][t6] + matrix[t7][t8] + matrix[t9][t10]
    gain -= (matrix[t4][t5] + matrix[t6][t7] + matrix[t8][t9] + matrix[t1][t10] + matrix[t2][t3])
    return gain


@nb.njit(cache=True)
def __get_tour(tour: np.ndarray, towns: np.ndarray) -> Tuple[bool, np.ndarray]:
    """ Собираем тур, для этого разбираем его на ребра
    tour: текущий список городов
    towns: полученные города
    return: получился ли корректный тур, тур
    """
    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = towns[0]

    t1t2, t3t4, t5t6, t7t8, t9t10 = \
        make_pair(t1, t2), make_pair(t3, t4), make_pair(t5, t6), make_pair(t7, t8), make_pair(t9, t10)
    t4t5, t6t7, t8t9, t1t10, t2t3 = \
        make_pair(t4, t5), make_pair(t6, t7), make_pair(t8, t9), make_pair(t1, t10), make_pair(t2, t3)

    edges, size = {(0, 0)}, len(tour)
    edges.clear()
    for i in range(size):
        edges.add(make_pair(tour[i - 1], tour[i]))

    edges = (edges - {t1t2, t3t4, t5t6, t7t8, t9t10}) | {t4t5, t6t7, t8t9, t1t10, t2t3}
    if len(edges) != size:  # добавили ребро, которое уже существет
        return False, tour

    successors, node = nb.typed.Dict.empty(nb.int64, nb.int64), 0
    while len(edges) > 0:
        x, y = 0, 0
        for x, y in edges:
            if x == node:
                successors[node] = y
                node = y
                break
            elif y == node:
                successors[node] = x
                node = x
                break
        edges.remove((x, y))

    idx, runner = 0, successors[0]
    visited = np.zeros(size, dtype=nb.boolean)
    _tour = np.zeros(size, dtype=nb.int64)

    while idx < size:
        if visited[runner]:
            break
        visited[runner] = True
        _tour[idx] = runner
        runner = successors[runner]
        idx += 1

    if idx < size:  # означает, что у нас цикл оказался
        return False, tour
    return True, _tour
