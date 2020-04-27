from __future__ import annotations

from collections import deque, defaultdict
from math import sqrt
from typing import Dict

import numpy as np
from numba import njit

from src.structures.one_tree import one_tree_topology


def _sort_topologically(first: int, topology: Dict):
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


def alpha_matrix(adjacency: np.ndarray) -> np.ndarray:
    size = adjacency.shape[0]
    f, s, topology = one_tree_topology(adjacency)
    matrix = np.zeros(shape=adjacency.shape)

    ordered_nodes = _sort_topologically(1, topology)

    for i in range(1, size):
        if i == f[0] or i == s[0]:
            continue
        matrix[0][i] = matrix[i][0] = s[1]

    for i in range(1, size - 1):
        for j in range(i + 1, size):
            idx = ordered_nodes[i - 1]
            idy = ordered_nodes[j - 1]

            matrix[idx][idy] = matrix[idy][idx] = max(matrix[idx][topology[idy]], adjacency[idy][topology[idy]])

    return adjacency - matrix


@njit
def adjacency_matrix(points: np.ndarray) -> np.ndarray:
    size = points.shape[0]
    matrix = np.zeros(shape=(size, size))
    for idx in range(0, size):
        for idy in range(idx + 1, size):
            distance = sqrt((points[idy][0] - points[idx][0]) ** 2 + (points[idy][1] - points[idx][1]) ** 2)
            matrix[idx][idy] = matrix[idy][idx] = distance
    return matrix


def savings_matrix(adjacency: np.ndarray, point: int) -> np.ndarray:
    matrix = np.zeros(shape=adjacency.shape)
    for idx in range(0, matrix.shape[0]):
        for idy in range(idx + 1, matrix.shape[0]):
            savings = adjacency[point][idx] + matrix[point][idy] - matrix[idx][idy]
            matrix[idx][idy] = matrix[idy][idx] = savings
    return matrix
