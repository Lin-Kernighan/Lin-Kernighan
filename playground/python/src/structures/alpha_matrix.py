from typing import List

from src.structures.matrix import Matrix
from src.structures.one_tree import OneTree


class AlphaMatrix(Matrix):

    def __init__(self, weight_matrix: List[List[float]], optimal: OneTree) -> None:
        self.matrix = []
        self.length = len(weight_matrix)

        for idx in range(self.length):
            self.matrix.append([0] * self.length)

        for idx in range(0, self.length):
            for idy in range(idx + 1, self.length):
                alpha_nearness = OneTree(weight_matrix, with_edge=(idx, idy)).total_price - optimal.total_price
                self.matrix[idx][idy] = self.matrix[idy][idx] = alpha_nearness
