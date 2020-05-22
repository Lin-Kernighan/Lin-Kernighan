from typing import Tuple, Dict

import numba as nb
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from src.algorithms.structures.heap import Heap
from src.algorithms.utils.utils import make_pair


def one_tree(adjacency_matrix: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """ MST( все точки кроме нулевой ) + два ребра от нулевой вершины
    return: длина графа, два массива в формате начало (src array) - конец (dst array)
    """
    # noinspection PyTypeChecker
    coo: coo_matrix = minimum_spanning_tree(adjacency_matrix[1:, 1:]).tocoo()
    src, dst, temp = coo.col + 1, coo.row + 1, coo.data.sum()
    f_node, s_node, f_min, s_min = __search(adjacency_matrix)
    return temp + f_min + s_min, np.append(src, [0, 0]), np.append(dst, [f_node, s_node])


@nb.njit(cache=True)
def __search(adjacency_matrix: np.ndarray) -> tuple:
    f_node, s_node, f_min, s_min = -1, -1, np.inf, np.inf
    for index, price in enumerate(adjacency_matrix[0]):
        if 0 == index or not price > 0:
            continue
        if price < f_min:
            s_node, s_min = f_node, f_min
            f_node, f_min = index, price
        elif price < s_min:
            s_node, s_min = index, price
    return f_node, s_node, f_min, s_min


def one_tree_topology(adjacency_matrix: np.ndarray) -> Tuple[float, tuple, tuple, set, Dict[int, int]]:
    """ build One tree
    return: длина one tree; минимальное ребро; пред минмальное ребро; set ребер; словарь son -> dad для MST
    """
    size, k, length = adjacency_matrix.shape[0], 0, 0.0
    topology: Dict[int, int] = {}
    visited = np.zeros(size, dtype=bool)
    heap, edges = Heap(), set()

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
        edges.add(make_pair(dst, src))
        add(dst)
        length += adjacency_matrix[src][dst]
        k += 1

    f_node, s_node, f_min, s_min = __search(adjacency_matrix)

    edges.add(make_pair(0, f_node))
    edges.add(make_pair(0, s_node))
    return length + f_min + s_min, (f_node, f_min), (s_node, s_min), edges, topology
