from sys import maxsize
from typing import List

from src.graph import Edge
from src.heap import StdHeap


class OneTree:
    edges: List[Edge]
    total_price: int
    length: int

    def __init__(self, weight_matrix: List[List[float]], node: int, with_edge: List[int] = None) -> None:
        """ One Tree for algorithms
        node: node for build one-tree for alpha nearness
        with_edge: pre-added edge to mst tree
        """
        self.length = len(weight_matrix)
        self.edges: List[Edge] = [Edge(0, 0, 0)] * self.length  # for n - 1 edges + one edge from node
        self.total_price = 0

        heap = StdHeap()
        checklist: List[bool] = [False] * self.length  # for checking before adding last edge in one-tree
        visited: List[bool] = [False] * self.length  # for searching not visited nodes in Prim's algorithm

        def add(idx: int, without: int = None):
            """ Add Edges from new node to heap
            """
            visited[idx] = True
            for idy, price in enumerate(weight_matrix[idx]):
                if price == 0 or visited[idy] or without == idy:
                    continue
                heap.push(Edge(price, idx, idy))

        k = 0
        if with_edge is not None:  # add additional edge
            x, y = with_edge
            self.edges[0] = Edge(weight_matrix[x][y], x, y)
            self.__check_edge(node, x, y, checklist)
            self.total_price += weight_matrix[x][y]
            visited[y] = visited[x] = True
            add(x, y)  # add all edges from x without y
            add(y)  # add all edges from y
            k += 1
        else:  # or just start
            add(0)

        while k < self.length - 1:  # another
            was, new_edge = True, None
            while was:
                new_edge = heap.pop()
                was = visited[new_edge.dst]  # check dst node
            self.edges[k] = new_edge
            self.__check_edge(node, new_edge.dst, new_edge.src, checklist)
            self.total_price += new_edge.price
            add(new_edge.dst)
            k += 1

        self.edges[-1] = self.__add_last_edge(weight_matrix[node], node, checklist)

    @staticmethod
    def __check_edge(node: int, x: int, y: int, checklist: List[bool]) -> None:
        if x == node:
            checklist[y] = True
        elif y == node:
            checklist[x] = True

    @staticmethod
    def __add_last_edge(prices: List[float], node: int, checklist: List[bool]) -> Edge:
        n_node, min_edge = -1, maxsize
        for index, price in enumerate(prices):
            if index == node:
                continue

            if price < min_edge and not checklist[index]:
                n_node, min_edge = index, price
        if n_node == -1 or min_edge == maxsize:
            raise Exception('Bad one-tree, not found last edge')
        return Edge(min_edge, node, n_node)

    def __len__(self) -> int:
        return self.total_price

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self.edges};\ntotal_price={self.total_price}'
