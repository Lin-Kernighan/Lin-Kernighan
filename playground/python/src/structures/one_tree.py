from __future__ import annotations

from dataclasses import field, dataclass
from typing import Tuple, Dict, List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from src.structures.heap import Heap


@dataclass
class OneTree:
    length: int
    total_price: float = field(default=0.0, init=False)
    edges: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        self.edges: List[Tuple[int, int]] = [(0, 0)] * self.length

    @staticmethod
    def build(adjacency_matrix, with_edge: Tuple[int, int] = None) -> OneTree:
        """ One Tree for algorithms
        node: node for build one-tree for alpha nearness
        with_edge: pre-added edge to mst tree
        """
        length = len(adjacency_matrix)
        tree = OneTree(length)
        tree.total_price = 0

        heap = Heap()
        visited: List[bool] = [False] * length  # for searching not visited nodes in Prim's algorithm

        def add(idx: int, without: int = None):
            """ Add Edges from new node to heap
            """
            visited[idx] = True
            for idy, price in enumerate(adjacency_matrix[idx]):
                if price == 0 or visited[idy] or without == idy or idy == 0:
                    continue
                heap.push((price, idx, idy))

        k = 0
        if with_edge is not None:  # add additional edge
            x, y = with_edge
            tree.edges[0] = (x, y)
            tree.total_price += adjacency_matrix[x][y]
            visited[y] = visited[x] = True
            add(x, y)  # add all edges from x without y
            add(y)  # add all edges from y
            k += 1
        else:  # or just start
            add(1)

        while k < length - 2:  # another
            was, src, dst, price = True, 0, 0, 0.0
            while was:
                price, src, dst = heap.pop()
                was = visited[dst]  # check dst node
            tree.edges[k] = (src, dst)
            tree.total_price += price
            add(dst)
            k += 1

        f_node, s_node, f_min, s_min = -1, -1, float('+inf'), float('+inf')
        for index, price in enumerate(adjacency_matrix[0]):
            if 0 == index or not price > 0:
                continue
            if price < f_min:
                s_node, s_min = f_node, f_min
                f_node, f_min = index, price
            elif price < s_min:
                s_node, s_min = index, price

        tree.total_price = tree.total_price + s_min + f_min
        tree.edges[-1] = (0, f_node)
        tree.edges[-2] = (0, s_node)

        return tree


def one_tree(adjacency_matrix: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    # noinspection PyTypeChecker
    mst: csr_matrix = minimum_spanning_tree(adjacency_matrix[1:, 1:])
    coo = mst.tocoo()
    src, dst, temp = coo.col + 1, coo.row + 1, coo.data.sum()

    f_node, s_node, f_min, s_min = -1, -1, float('+inf'), float('+inf')
    for index, price in enumerate(adjacency_matrix[0]):
        if 0 == index or not price > 0:
            continue
        if price < f_min:
            s_node, s_min = f_node, f_min
            f_node, f_min = index, price
        elif price < s_min:
            s_node, s_min = index, price

    return temp + f_min + s_min, np.append(src, [0, 0]), np.append(dst, [f_node, s_node])


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
    #         s_node, s_min = f_node, f_min
    #         f_node, f_min = index, price
    #     elif price < s_min:
    #         s_node, s_min = index, price
    #
    # print(f_node, s_node)
    # topology[f_node] = 0
    # topology[s_node] = 0
    # total_price = total_price + f_min + s_min
    return total_price, topology
