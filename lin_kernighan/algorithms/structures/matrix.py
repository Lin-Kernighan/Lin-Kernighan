from collections import deque, defaultdict
from math import sqrt
from typing import Dict

import numpy as np
from numba import njit


def _sort_topologically(first: int, topology: Dict[int, int]):
    """ node -> dad; in fact this is Depth-first search """
    ancestors = defaultdict(list)
    for key, pred in topology.items():
        ancestors[pred] += [key]

    res = []
    deq = deque()
    deq.append(first)

    while deq:
        node = deq.popleft()
        res += [node]
        deq.extend(ancestors[node])

    return res


def alpha_matrix(adjacency: np.ndarray, f: tuple, s: tuple, topology: Dict[int, int]) -> np.ndarray:
    """ Альфа матрица - изменение длины one tree, если пред добавить другое ребро
    adjacency: матрица весов
    f: минимальное ребро от 0 вершины (length, num)
    s: пред минимальное ребро от 0 вершины (length, num)
    topology: словарь son -> dad для вершин в MST графе
    """
    size = adjacency.shape[0]
    matrix = np.zeros(shape=adjacency.shape)

    ordered_nodes = _sort_topologically(1, topology)

    for i in range(1, size):
        if i == f[0]:
            matrix[0][i] = matrix[i][0] = f[1]
        if i == s[0]:
            matrix[0][i] = matrix[i][0] = s[1]

    for i in range(1, size - 1):
        for j in range(i + 1, size):
            idx = ordered_nodes[i - 1]
            idy = ordered_nodes[j - 1]

            matrix[idx][idy] = matrix[idy][idx] = max(matrix[idx][topology[idy]], adjacency[idy][topology[idy]])

    return adjacency - matrix


@njit(parallel=True, cache=True)
def adjacency_matrix(points: np.ndarray) -> np.ndarray:
    """ Матрица смежности """
    size = points.shape[0]
    matrix = np.zeros(shape=(size, size))
    for idx in range(0, size):
        for idy in range(idx + 1, size):
            distance = sqrt((points[idy][0] - points[idx][0]) ** 2 + (points[idy][1] - points[idx][1]) ** 2)
            matrix[idx][idy] = matrix[idy][idx] = distance
    return matrix


@njit(parallel=True, cache=True)
def savings_matrix(adjacency: np.ndarray, point: int) -> np.ndarray:
    """ Матрица savings для Clarke-Wright """
    matrix = np.zeros(shape=adjacency.shape)
    for idx in range(0, matrix.shape[0]):
        for idy in range(idx + 1, matrix.shape[0]):
            savings = adjacency[point][idx] + matrix[point][idy] - matrix[idx][idy]
            matrix[idx][idy] = matrix[idy][idx] = savings
    return matrix
