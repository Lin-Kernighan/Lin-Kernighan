from __future__ import annotations

from dataclasses import field, dataclass
from sys import maxsize
from typing import Tuple, Dict, List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from src.structures.heap import Heap


@dataclass(order=True)
class Edge:
    price: float
    src: int
    dst: int

    def __str__(self) -> str:
        return f'{self.src}->{self.dst}'

    def __repr__(self) -> str:
        return str(self)


@dataclass
class OneTree:
    length: int
    total_price: float = field(default=0.0, init=False)
    edges: List[Edge] = field(default_factory=list)

    def __post_init__(self):
        self.edges: List[Edge] = [Edge(0, 0, 0)] * self.length

    @staticmethod
    def build(adjacency_matrix, node: int = 0, with_edge: Tuple[int, int] = None) -> OneTree:
        """ One Tree for algorithms
        node: node for build one-tree for alpha nearness
        with_edge: pre-added edge to mst tree
        """
        length = len(adjacency_matrix)
        tree = OneTree(length)
        # for n - 1 edges + one edge from node
        tree.total_price = 0

        heap = Heap()
        checklist: List[bool] = [False] * length  # for checking before adding last edge in one-tree
        visited: List[bool] = [False] * length  # for searching not visited nodes in Prim's algorithm

        def add(idx: int, without: int = None):
            """ Add Edges from new node to heap
            """
            visited[idx] = True
            for idy, price in enumerate(adjacency_matrix[idx]):
                if price == 0 or visited[idy] or without == idy:
                    continue
                heap.push(Edge(price, idx, idy))

        k = 0
        if with_edge is not None:  # add additional edge
            x, y = with_edge
            tree.edges[0] = Edge(adjacency_matrix[x][y], x, y)
            tree.__check_edge(node, x, y, checklist)
            tree.total_price += adjacency_matrix[x][y]
            visited[y] = visited[x] = True
            add(x, y)  # add all edges from x without y
            add(y)  # add all edges from y
            k += 1
        else:  # or just start
            add(0)

        while k < length - 1:  # another
            was, new_edge = True, None
            while was:
                new_edge = heap.pop()
                was = visited[new_edge.dst]  # check dst node
            tree.edges[k] = new_edge
            tree.__check_edge(node, new_edge.dst, new_edge.src, checklist)
            tree.total_price += new_edge.price
            add(new_edge.dst)
            k += 1

        tree.edges[-1] = tree.__add_last_edge(adjacency_matrix[node], node, checklist)
        tree.total_price += tree.edges[-1].price
        return tree

    @staticmethod
    def __check_edge(node: int, x: int, y: int, checklist: List[bool]) -> None:
        """ Check if one of node is Node... if that, mark it
        """
        if x == node:
            checklist[y] = True
        elif y == node:
            checklist[x] = True

    @staticmethod
    def __add_last_edge(prices: List[float], node: int, checklist: List[bool]) -> Edge:
        """ Add last edge, mst -> one tree
        """
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
        return self.length

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self.edges};\ntotal_price={self.total_price}'


def one_tree(adjacency_matrix: np.ndarray, node: int = 0) -> Tuple[float, np.ndarray, np.ndarray]:
    # noinspection PyTypeChecker
    mst: csr_matrix = minimum_spanning_tree(adjacency_matrix)
    coo = mst.tocoo()
    src, dst, temp = coo.col, coo.row, coo.data.sum()

    indexes = [dst[idx] for idx in np.where(src == node)[0]] + \
              [src[idx] for idx in np.where(dst == node)[0]]

    that, minimum = -1, float('inf')
    for idx, value in enumerate(adjacency_matrix[node]):
        if node == idx or not value > 0:
            continue
        if value < minimum and idx not in indexes:
            that, minimum = idx, value

    return temp + minimum, np.append(src, node), np.append(dst, that)


def one_tree_topology(adjacency_matrix: np.ndarray) -> Tuple[float, Dict[int, int]]:
    size = adjacency_matrix.shape[0]
    total_price, k = 0.0, 0
    topology: Dict[int, int] = {}
    visited = np.zeros(size, dtype=bool)
    heap = Heap()

    def add(idx: int):
        visited[idx] = True
        for idy, value in enumerate(adjacency_matrix[idx]):
            if not value > 0 or visited[idy] or idy == 0:
                continue
            heap.push((value, idx, idy))

    add(1)
    while k < size - 2:
        was, price, src, dst = True, 0.0, 0, 0
        while was:
            price, src, dst = heap.pop()
            was = visited[dst]
        topology[dst] = src
        total_price += price
        add(dst)
        k += 1

    # f_node, s_node, f_min, s_min = -1, -1, float('+inf'), float('+inf')
    # print(adjacency_matrix[0])
    # for index, price in enumerate(adjacency_matrix[0]):
    #     if index == 0:
    #         continue
    #     if price < f_min:
    #         f_node, f_min = index, price
    #     elif price < s_min:
    #         s_node, s_min = index, price
    #
    # print(f_node, s_node)
    # topology[f_node] = 0
    # topology[s_node] = 0
    # total_price = total_price + f_min + s_min
    return total_price, topology
