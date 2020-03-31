from dataclasses import dataclass

from src.heap import StdHeap
import numpy as np


@dataclass(order=True)
class Edge:
    price: int
    src: int
    dst: int

    def __str__(self):
        return f'{self.src}->{self.dst}'

    def __repr__(self):
        return str(self)


def prim_tree(matrix: np.ndarray):
    n, res = len(matrix), []
    heap = StdHeap()
    visited = [False] * n

    def visit(u):
        visited[u] = True

        for v in range(n):
            if not visited[v] and matrix[u][v] > 0:
                heap.push(Edge(matrix[u][v], u, v))

    visit(0)
    for i in range(n - 1):
        res += [heap.pop()]
        visit(res[-1].dst)

    return res


if __name__ == "__main__":
    print('Prim:\n', prim_tree(np.array([
        [0, 2, 0, 6, 0],
        [2, 0, 3, 8, 5],
        [0, 3, 0, 0, 7],
        [6, 8, 0, 0, 9],
        [0, 5, 7, 9, 0],
    ])))
