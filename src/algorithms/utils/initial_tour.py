from random import randrange
from sys import maxsize
from typing import Tuple, Set

import numba as nb
import numpy as np

from src.algorithms.two_opt import TwoOpt

Edge = Tuple[int, int]


def clarke_wright() -> Tuple[float, np.ndarray]:
    # TODO: Clarke-Wright
    pass


def popmusic() -> Tuple[float, np.ndarray]:
    # TODO: POPMUSIC
    pass


def fast_helsgaun(alpha_matrix: np.ndarray, adjacency_matrix: np.ndarray, best_solution: Set[Edge],
                  candidates: np.ndarray, excess: float) -> Tuple[float, np.ndarray]:
    """ Генерируем новый тур по рецепту Хельгауна, c постоптимизацей 2-opt
    alpha_matrix: альфа-матрица
    adjacency_matrix: матрица весов
    best_solution: лучший тур в виде ребер
    candidates: сгенерированные кандидаты для LKH
    excess: уровень по которому отсекаются кандидаты
    return: длина, список городов
    """
    length, tour = helsgaun(alpha_matrix, adjacency_matrix, best_solution, candidates, excess)
    return TwoOpt.just_improve(length, tour, adjacency_matrix)


@nb.njit(cache=True)
def greedy(matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """ Генерация начального тура жадным методом
    matrix: матрица весов
    return: длина, список городов
    """
    length = matrix.shape[0]
    start = previous = randrange(0, length)  # я ищу ребро из previous в search
    search, path, k = 0, 0.0, 0

    visited = np.zeros((length,), dtype=nb.int64)
    order = np.zeros((length,), dtype=nb.int64)
    visited[previous] = 1

    while k < length - 1:
        minimum, search = maxsize, -1
        for idx, price in enumerate(matrix[previous]):
            if idx != previous and minimum > price and visited[idx] == 0:
                minimum, search = price, idx
        visited[search] = 1
        path += minimum
        order[k] = previous
        previous = search
        k += 1

    path += matrix[search][start]
    order[-1] = search
    return path, order


@nb.njit(cache=True)
def helsgaun(alpha_matrix: np.ndarray, adjacency_matrix: np.ndarray, best_solution: Set[Edge],
             candidates: np.ndarray, excess: float) -> Tuple[float, np.ndarray]:
    """ Генерируем новый тур по рецепту Хельгауна
    alpha_matrix: альфа-матрица
    adjacency_matrix: матрица весов
    best_solution: лучший тур в виде ребер
    candidates: сгенерированные кандидаты для LKH
    excess: уровень по которому отсекаются кандидаты
    return: длина, список городов
    """
    size, k, length = alpha_matrix.shape[0], 0, 0.0
    previous = search = randrange(0, size)  # я ищу ребро из previous в search
    visited = np.zeros(size, dtype=nb.boolean)
    order = np.zeros(size, dtype=nb.int64)
    visited[previous] = True

    while k < size - 1:
        prices = alpha_matrix[previous]
        if (search := __zero_alpha(previous, prices, visited)) != -1:
            length += adjacency_matrix[previous][search]
        elif (search := __best_tour(previous, prices, excess, best_solution, visited)) != -1:
            length += adjacency_matrix[previous][search]
        elif (search := __get_candidate_set(previous, candidates, visited)) != -1:
            length += adjacency_matrix[previous][search]
        elif (search := __just_random(previous, prices, visited)) != -1:
            length += adjacency_matrix[previous][search]
        else:
            raise RuntimeError('Edge not found')

        order[k] = previous
        visited[search] = True
        previous = search
        k += 1

    order[-1] = search
    length += adjacency_matrix[order[0]][order[-1]]
    return length, order


@nb.njit(cache=True)
def __zero_alpha(previous: int, prices: np.ndarray, visited: np.ndarray) -> int:
    """ Перебираем все ребра с альфа-близостью равной нулю и выбираем из них рандомное """
    zeros = np.array([idx
                      for idx, alpha in enumerate(prices)
                      if alpha == 0 and not visited[idx] and idx != previous])
    if len(zeros) != 0:
        return np.random.choice(zeros)
    return -1


@nb.njit(cache=True)
def __search(edges: set, node: int) -> Set[Edge]:
    """ Ищем ребра с концом равным node """
    temp = {(0, 0)}
    temp.clear()
    for edge in edges:
        if node == edge[0] or node == edge[1]:
            temp.add(edge)
    return temp


@nb.njit(cache=True)
def __best_tour(previous: int, prices: np.ndarray, excess: float, best_solution: set, visited: np.ndarray) -> int:
    """ Ищем в лучшем туре """
    search = __search(best_solution, previous)
    node, alpha = -1, maxsize
    for edge in search:
        temp = edge[0] if edge[0] != previous else edge[1]
        if not visited[temp] and prices[temp] < alpha < excess:
            node, alpha = temp, prices[temp]
    return node if node != -1 else -1


@nb.njit(cache=True)
def __get_candidate_set(previous: int, candidates: np.ndarray, visited: np.ndarray) -> int:
    """ Рандомный из кандидатов """
    candidates = np.array([idx for idx in candidates[previous] if idx != -1 and not visited[idx]])
    if len(candidates) != 0:
        return np.random.choice(candidates)
    return -1


@nb.njit(cache=True)
def __just_random(previous: int, prices: np.ndarray, visited: np.ndarray) -> int:
    """ Хоть какой-нибудь """
    candidates = np.array([idx for idx, price in enumerate(prices) if not visited[idx] and previous != idx])
    if len(candidates) != 0:
        return np.random.choice(candidates)
    return -1
