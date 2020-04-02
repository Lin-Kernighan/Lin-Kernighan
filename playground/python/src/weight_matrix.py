from typing import List


class WeightMatrix:
    matrix: List[List[float]]
    length: int

    def __init__(self, points: List[List[float]]) -> None:
        self.matrix = []
        self.length = len(points)
        for idx in range(self.length):
            self.matrix.append([0] * self.length)

        for idx in range(0, self.length):
            for idy in range(idx + 1, self.length):
                distance = (points[idy][0] - points[idx][0]) ** 2 + (points[idy][1] - points[idx][1]) ** 2
                self.matrix[idx][idy] = self.matrix[idy][idx] = distance

    def __str__(self):
        string = ''
        for s in self.matrix:
            for elem in s:
                string += f'{elem:0.2f}\t'
            string += '\n'
        return string

    def __len__(self) -> int:
        return self.length

    def __repr__(self):
        return str(self)

    def __getitem__(self, index):
        return self.matrix[index]
