from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import List, Tuple

from src.structures.one_tree import OneTree


@dataclass
class Matrix:
    dimension: int
    matrix: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        for idx in range(self.dimension):
            self.matrix.append([0] * self.dimension)

    @staticmethod
    def alpha_matrix(weight_matrix: Matrix, optimal: OneTree) -> Matrix:
        alpha_matrix = Matrix(len(weight_matrix))

        for idx in range(0, alpha_matrix.dimension):
            for idy in range(idx + 1, alpha_matrix.dimension):
                alpha_nearness = OneTree.build(weight_matrix, with_edge=(idx, idy)).total_price - optimal.total_price
                alpha_matrix.matrix[idx][idy] = alpha_matrix.matrix[idy][idx] = alpha_nearness
        return alpha_matrix

    @staticmethod
    def weight_matrix(points: List[Tuple[float, float]]) -> Matrix:
        weight_matrix = Matrix(len(points))

        for idx in range(0, weight_matrix.dimension):
            for idy in range(idx + 1, weight_matrix.dimension):
                distance = sqrt((points[idy][0] - points[idx][0]) ** 2 + (points[idy][1] - points[idx][1]) ** 2)
                weight_matrix.matrix[idx][idy] = weight_matrix.matrix[idy][idx] = distance
        return weight_matrix

    def __str__(self):
        string = ''
        for s in self.matrix:
            for elem in s:
                string += f'{elem:0.2f}\t'
            string += '\n'
        return string

    def __len__(self) -> int:
        return self.dimension

    def __repr__(self):
        return str(self)

    def __getitem__(self, index: int) -> List[float]:
        return self.matrix[index]
