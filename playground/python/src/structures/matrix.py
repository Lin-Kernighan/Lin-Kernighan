from __future__ import annotations

from math import sqrt
from typing import List, Tuple

from numpy import ndarray, zeros

from src.structures.one_tree import OneTree

Point = Tuple[float, float]


class Matrix:

    @staticmethod
    def alpha_matrix(adjacency_matrix: ndarray, optimal: OneTree) -> ndarray:
        alpha_matrix = zeros(shape=adjacency_matrix.shape)
        # TODO: optimize this shit
        for idx in range(0, alpha_matrix.shape[0]):
            for idy in range(idx + 1, alpha_matrix.shape[0]):
                alpha_nearness = OneTree.build(adjacency_matrix, with_edge=(idx, idy)).total_price - optimal.total_price
                alpha_matrix[idx][idy] = alpha_matrix[idy][idx] = alpha_nearness
        return alpha_matrix

    @staticmethod
    def adjacency_matrix(points: List[Point]) -> ndarray:
        size = len(points)
        adjacency_matrix = zeros(shape=(size, size))
        for idx in range(0, size):
            for idy in range(idx + 1, size):
                distance = sqrt((points[idy][0] - points[idx][0]) ** 2 + (points[idy][1] - points[idx][1]) ** 2)
                adjacency_matrix[idx][idy] = adjacency_matrix[idy][idx] = distance
        return adjacency_matrix

    @staticmethod
    def savings_matrix(adjacency_matrix: ndarray, point: int) -> ndarray:
        savings_matrix = zeros(shape=adjacency_matrix.shape)
        for idx in range(0, savings_matrix.shape[0]):
            for idy in range(idx + 1, savings_matrix.shape[0]):
                savings = adjacency_matrix[point][idx] + adjacency_matrix[point][idy] - adjacency_matrix[idx][idy]
                savings_matrix[idx][idy] = savings_matrix[idy][idx] = savings
        return savings_matrix
