from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt

from src.heap import StdHeap
from src.weight_matrix import WeightMatrix


@dataclass(order=True)
class Edge:
    price: float
    src: int
    dst: int

    def __str__(self) -> str:
        return f'{self.src}->{self.dst}'

    def __repr__(self) -> str:
        return str(self)


class Graph:
    edges: List[Edge]
    nodes: List[List[float]]
    c: WeightMatrix
    length: int

    def __init__(self, points: List[List[float]]) -> None:
        self.nodes = points
        self.c = WeightMatrix(points)
        self.edges = []
        self.length = len(points)

    def prim_tree(self) -> None:
        heap = StdHeap()
        visited = [False] * self.length

        def visit(u):
            visited[u] = True

            for v in range(self.length):
                if not visited[v] and self.c[u][v] > 0:
                    heap.push(Edge(self.c[u][v], u, v))

        visit(0)
        for i in range(self.length - 1):
            self.edges += [heap.pop()]
            visit(self.edges[-1].dst)

    def draw(self):
        for edge in self.edges:
            [x1, y1] = self.nodes[edge.dst]
            [x2, y2] = self.nodes[edge.src]
            plt.plot([x1, x2], [y1, y2], linewidth=1, color='r')

        for idx, node in enumerate(self.nodes):
            plt.annotate(f'{idx}:({node[0]},{node[1]})', node, size=10)
        plt.show()
