from __future__ import annotations

from math import sqrt

from numba import njit
from numpy import ndarray, zeros

from src.structures.one_tree import OneTree


def alpha_matrix(adjacency: ndarray, optimal: OneTree) -> ndarray:
    matrix = zeros(shape=adjacency.shape)
    # TODO: optimize this shit
    for idx in range(0, matrix.shape[0]):
        for idy in range(idx + 1, matrix.shape[0]):
            alpha_nearness = OneTree.build(adjacency, with_edge=(idx, idy)).total_price - optimal.total_price
            matrix[idx][idy] = matrix[idy][idx] = alpha_nearness
    return matrix


@njit
def adjacency_matrix(points: ndarray) -> ndarray:
    size = points.shape[0]
    matrix = zeros(shape=(size, size))
    for idx in range(0, size):
        for idy in range(idx + 1, size):
            distance = sqrt((points[idy][0] - points[idx][0]) ** 2 + (points[idy][1] - points[idx][1]) ** 2)
            matrix[idx][idy] = matrix[idy][idx] = distance
    return matrix


def savings_matrix(adjacency: ndarray, point: int) -> ndarray:
    matrix = zeros(shape=adjacency.shape)
    for idx in range(0, matrix.shape[0]):
        for idy in range(idx + 1, matrix.shape[0]):
            savings = adjacency[point][idx] + matrix[point][idy] - matrix[idx][idy]
            matrix[idx][idy] = matrix[idy][idx] = savings
    return matrix
