from dataclasses import dataclass, field
from typing import Tuple, Set

import numba as nb
import numpy as np

from src.structures.tour.abc_tour import AbcTour
from src.utils import make_pair

Edge = Tuple[int, int]
Node = int


@nb.njit
def generate(size: int, edges: Set[Edge], broken: Set[Edge], joined: Set[Edge]) -> np.ndarray:
    """ Создаем новый тур, а потом проверяем его на целостность и наличие циклов
    broken: удаляемые ребра
    joined: добавляемые ребра
    """
    edges = (edges - broken) | joined
    # If we do not have enough edges, we cannot form a tour -- should not
    if len(edges) < size:
        return np.zeros(1, dtype=nb.int64)

    successors = {}
    node = 0

    # Build the list of successors
    while len(edges) > 0:
        i = j = 0
        for i, j in edges:
            if i == node:
                successors[node] = j
                node = j
                break
            elif j == node:
                successors[node] = i
                node = i
                break
        edges.remove((i, j))

    # Similarly, if not every node has a successor, this can not work
    if len(successors) < size:
        return np.zeros(1, dtype=nb.int64)

    successor, k = successors[0], 1
    new_tour = np.zeros(size, dtype=nb.int64)
    visited = set(new_tour)

    # If we already encountered a node it means we have a loop
    while successor not in visited:
        visited.add(successor)
        new_tour[k] = successor
        successor = successors[successor]
        k += 1

    # If we visited all nodes without a loop we have a tour
    return new_tour


@dataclass
class ListTour(AbcTour):
    tour: np.ndarray
    size: int = field(init=False)
    edges: Set[Edge] = field(init=False)

    def __post_init__(self):
        self.size = len(self.tour)
        self.edges = set()
        for i in range(self.size):
            self.edges.add(make_pair(self.tour[i - 1], self.tour[i]))

    def __len__(self):
        """ Кол-во вершин """
        return self.size

    def __getitem__(self, index: int) -> Node:
        """ Вершина по номеру в туре """
        return self.tour[index % self.size]

    def __contains__(self, edge: Edge) -> bool:
        """ Наличие ребра в туре """
        return edge in self.edges

    def index(self, node: Node) -> int:
        """ Номер вершины в туре """
        return np.where(self.tour == node)[0][0]

    def around(self, node: Node) -> Tuple[Node, Node]:
        """ Предыдущая вершина и следующая текущей веришны """
        index = self.index(node)
        return self[index - 1], self[index + 1]

    def successor(self, index: int) -> Node:
        """ Следующий """
        return self[index + 1]

    def predecessor(self, index: int) -> Node:
        """ Предыдущий """
        return self[index - 1]

    def between(self, start: Node, end: Node, search: Node) -> bool:
        """ Находится ли вершина search между вершиной start и вершиной end """
        start_index, end_index, search_index = self.index(start), self.index(end), self.index(search)

        if start_index < end_index:  # [ ... start -> ... search ... <- end ... ]
            if start_index < search_index < end_index:
                return True
        else:  # [ ? ... <- end ... start -> ... ? ]
            if 0 <= search_index < end_index or start_index < search_index < self.size:
                return True
        return False

    def generate(self, broken: Set[Edge], joined: Set[Edge]) -> np.ndarray:
        """ Создаем новый тур, а потом проверяем его на целостность и наличие циклов
        broken: удаляемые ребра
        joined: добавляемые ребра
        """
        # New edges: old edges minus broken, plus joined
        return generate(self.size, self.edges, broken, joined)

    def reverse(self, start: int, end: int) -> None:
        """ Переворот куска тура """
        # TODO: доделать
