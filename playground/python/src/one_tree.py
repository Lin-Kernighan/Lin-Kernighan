from sys import maxsize
from typing import List

from src.graph import Edge
from src.heap import StdHeap


class OneTree:
    edges: List[Edge]
    total_price: int
    length: int

    def __init__(self, weight_matrix: List[List[float]], node: int, to_node: int = None) -> None:
        """ One Tree for algorithms
        node: node for build one-tree for alpha nearness
        to_node: with additional edge to_node from node
        """
        self.length = len(weight_matrix)
        self.edges: List[Edge] = [Edge(0, 0, 0)] * self.length  # for n - 1 edges + two edges from node
        self.total_price = 0

        heap = StdHeap()
        visited: List[bool] = [False] * self.length
        visited[node] = True  # without node

        def add(idx: int, without: int = None):
            """ Add Edges from new node to heap
            """
            visited[idx] = True
            for idy, price in enumerate(weight_matrix[idx]):
                if price == 0 or visited[idy] or without == idy:
                    continue
                heap.push(Edge(price, idx, idy))

        if node != 0:  # start
            add(0)
        else:
            add(1)

        k = 0
        while k < self.length - 2:  # without two edge from node
            was, new_edge = True, None
            while was:
                new_edge = heap.pop()
                was = visited[new_edge.dst]  # check dst node
            self.edges[k] = new_edge
            self.total_price += new_edge.price
            add(new_edge.dst)
            k += 1

        if to_node is None:  # add two edge from node
            self.__add_two_minimum(node, weight_matrix)  # if only two minimal node
        else:
            self.edges[-2] = Edge(weight_matrix[node][to_node], node, to_node)
            self.total_price += weight_matrix[node][to_node]
            self.__add_one_minimum(node, weight_matrix, to_node)

    def __add_one_minimum(self, node: int, weight_matrix: List[List[float]], to_node: int) -> None:
        """ Find and add minimum from node
        to_node: ignore edge from node - to_node? it was added previous
        """
        index, minimum = -1, maxsize
        for idx in range(self.length):
            if node == idx or to_node == idx:
                continue
            if minimum > weight_matrix[node][idx]:
                index, minimum = idx, weight_matrix[node][idx]
        self.edges[-1] = Edge(minimum, node, index)
        self.total_price += minimum

    def __add_two_minimum(self, node: int, weight_matrix: List[List[float]]) -> None:
        """ Find two minimal edges from node and add them
        """
        first, first_min = -1, maxsize
        second, second_min = -1, maxsize
        for idx in range(self.length):  # find two minimum edge
            if idx == node:
                continue
            if first_min > weight_matrix[node][idx]:
                second, second_min = first, first_min  # exchange for new first minimum
                first, first_min = idx, weight_matrix[node][idx]
            elif second_min > weight_matrix[node][idx]:
                second, second_min = idx, weight_matrix[node][idx]
            else:
                continue
        self.edges[-2] = Edge(first_min, node, first)  # add them
        self.edges[-1] = Edge(second_min, node, second)
        self.total_price += first_min
        self.total_price += second_min

    def __len__(self) -> int:
        return self.total_price

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self.edges};\ntotal_price={self.total_price}'
