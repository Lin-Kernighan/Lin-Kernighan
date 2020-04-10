from math import sqrt
from typing import List, Tuple

from src.structures.matrix import Matrix


class WeightMatrix(Matrix):

    def __init__(self, points: List[Tuple[float, float]]) -> None:
        self.matrix = []
        self.length = len(points)
        for idx in range(self.length):
            self.matrix.append([0] * self.length)

        for idx in range(0, self.length):
            for idy in range(idx + 1, self.length):
                distance = sqrt((points[idy][0] - points[idx][0]) ** 2 + (points[idy][1] - points[idx][1]) ** 2)
                self.matrix[idx][idy] = self.matrix[idy][idx] = distance
