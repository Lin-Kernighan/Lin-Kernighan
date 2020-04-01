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
        self.length = len(points)
        self.edges = [Edge(0, 0, 0)] * (self.length - 1)

    def prim_tree(self, with_edge: List[int] = None) -> float:
        """ Prim's algorithm
        with_edge: build MST with edge for calc alpha nearness
        """

        def add(idx: int, without: int = None):
            """ Add Edges from new node to heap
            """
            visited[idx] = True
            for idy, price in enumerate(self.c[idx]):
                if price == 0 or visited[idy] or without == idy:
                    continue
                heap.push(Edge(price, idx, idy))

        heap = StdHeap()
        visited = [False] * self.length

        total_price = 0
        k = 0
        if with_edge is not None:
            x, y = with_edge
            self.edges[0] = Edge(self.c[x][y], x, y)
            total_price += self.c[x][y]
            visited[y] = visited[x] = True
            add(x, y)  # add all edges from x without y
            add(y)  # add all edges from y
            k += 1
        else:
            add(0)

        while k < self.length - 1:
            was, new_edge = True, None
            while was:
                new_edge = heap.pop()
                was = visited[new_edge.dst]  # check dst node
            self.edges[k] = new_edge
            total_price += new_edge.price
            add(new_edge.dst)
            k += 1
        return total_price

    def draw(self):
        for edge in self.edges:
            [x1, y1] = self.nodes[edge.dst]
            [x2, y2] = self.nodes[edge.src]
            plt.plot([x1, x2], [y1, y2], linewidth=1, color='r')

        for idx, node in enumerate(self.nodes):
            plt.annotate(f'{idx}:({node[0]},{node[1]})', node, size=9)
        plt.show()
