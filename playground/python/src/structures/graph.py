from dataclasses import dataclass, field
from typing import Set, Tuple

Edge = Tuple[int, int]


@dataclass
class PoolEdges:
    """ Хранилище множества ребер """
    edges: Set[Edge] = field(default_factory=set)

    def add(self, edge: Edge) -> None:
        """ Докидываем еще одно ребро в правильном порядке """
        idx, idy = edge
        if idx == idy:
            raise RuntimeError(f'idx == idy in edge:{edge}')

        temp = (idx, idy) if idx < idy else (idy, idx)
        if temp in self.edges:
            raise RuntimeError(f'edge:{edge} is already in pool')
        else:
            self.edges.add(temp)

    def search(self, node: int) -> Set[Edge]:
        """ Ищем ребра с концом равным node """
        temp = set()
        for edge in self.edges:
            if node == edge[0] or node == edge[1]:
                temp.add(edge)
        return temp

    def __contains__(self, item: Edge) -> bool:
        idx, idy = item
        temp = (idx, idy) if idx < idy else (idy, idx)
        return True if temp in self.edges else False


class Graph(PoolEdges):
    total_length: float = 0.0

    def add(self, edge: Edge, price: float = 0) -> None:
        """ Докидываем ребро, увеличиваем длину """
        idx, idy = edge
        if idx == idy:
            raise RuntimeError(f'idx == idy in edge:{edge}')

        temp = (idx, idy) if idx < idy else (idy, idx)
        if edge in self.edges:
            raise RuntimeError(f'edge:{edge} is already in pool')
        else:
            self.edges.add(temp)
            self.total_length += price
