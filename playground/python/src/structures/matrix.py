from __future__ import annotations

from math import sqrt

import numpy as np
from numba import njit

from src.structures.one_tree import one_tree_topology, OneTree, one_tree
from src.utils import print_matrix


def alpha_matrix(adjacency: np.ndarray) -> np.ndarray:
    total_price, _, _ = one_tree(adjacency)

    matrix = np.zeros(shape=adjacency.shape)
    # TODO: optimize this shit
    for idx in range(0, matrix.shape[0]):
        for idy in range(idx + 1, matrix.shape[0]):
            alpha_nearness = OneTree.build(adjacency, with_edge=(idx, idy)).total_price - total_price
            matrix[idx][idy] = matrix[idy][idx] = alpha_nearness

    print_matrix(matrix)
    return matrix


def betta_matrix(adjacency: np.ndarray) -> np.ndarray:
    size = adjacency.shape[0]
    _, topology = one_tree_topology(adjacency)
    matrix = np.zeros(shape=adjacency.shape)
    print(topology)

    for idx in range(1, size - 1):
        for idy in range(idx + 1, size):
            matrix[idx][idy] = matrix[idy][idx] = max(matrix[idx][topology[idy]], adjacency[idy][topology[idy]])

    print_matrix(matrix)
    print_matrix(adjacency - matrix)

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
