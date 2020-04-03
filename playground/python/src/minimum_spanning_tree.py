from typing import List

from src.heap import StdHeap
from src.graph import Edge


class MinimumSpanningTree:
    edges: List[Edge]
    total_price: int
    length: int

    def __init__(self, weight_matrix: List[List[float]], with_edge: List[int] = None) -> None:
        """ Prim's algorithm
        with_edge: build MST with edge for calc alpha nearness
        """
        self.length = len(weight_matrix)
        self.edges: List[Edge] = [Edge(0, 0, 0)] * (self.length - 1)
        self.total_price = 0
        heap = StdHeap()
        visited: List[bool] = [False] * self.length

        def add(idx: int, without: int = None):
            """ Add Edges from new node to heap
            """
            visited[idx] = True
            for idy, price in enumerate(weight_matrix[idx]):
                if price == 0 or visited[idy] or without == idy:
                    continue
                heap.push(Edge(price, idx, idy))

        k = 0
        if with_edge is not None:
            x, y = with_edge
            self.edges[0] = Edge(weight_matrix[x][y], x, y)
            self.total_price += weight_matrix[x][y]
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
            self.total_price += new_edge.price
            add(new_edge.dst)
            k += 1

    def __len__(self) -> int:
        return self.total_price

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self.edges};\ntotal_price={self.total_price}'
