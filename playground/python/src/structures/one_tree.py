from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from src.structures.heap import Heap


def one_tree(adjacency_matrix: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """ MST( все точки кроме нулевой ) + два ребра от нулевой вершины
    return: длина графа, два массива в формате начало (src array) - конец (dst array)
    """
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


def one_tree_topology(adjacency_matrix: np.ndarray) -> Tuple[float, tuple, tuple, Dict[int, int]]:
    """ One tree
    return: два минимальных ребра от вершины 0 (f < s) + словарь son -> dad для MST
    """
    size, k, length = adjacency_matrix.shape[0], 0, 0.0
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
        add(dst)
        length += adjacency_matrix[src][dst]
        k += 1

    f_node, s_node, f_min, s_min = -1, -1, float('+inf'), float('+inf')
    for index, price in enumerate(adjacency_matrix[0]):
        if index == 0:
            continue
        if price < f_min:
            s_node, s_min = f_node, f_min
            f_node, f_min = index, price
        elif price < s_min:
            s_node, s_min = index, price

    return length + f_min + s_min, (f_node, f_min), (s_node, s_min), topology
