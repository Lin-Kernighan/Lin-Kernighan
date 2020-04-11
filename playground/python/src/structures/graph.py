from dataclasses import dataclass, field
from typing import Set, Tuple


@dataclass
class Edge:
    price: float
    src: int
    dst: int

    def __str__(self) -> str:
        return f'{self.src}->{self.dst}'

    def __repr__(self) -> str:
        return str(self)


@dataclass
class PoolEdges:
    """ Хранилище множества ребер """
    edges: Set[Tuple[int, int]] = field(default_factory=set)

    def add(self, edge: Tuple[int, int]) -> None:
        """ Докидываем еще одно ребро в правильном порядке """
        idx, idy = edge
        if idx == idy:
            raise RuntimeError(f'idx == idy in edge:{edge}')

        temp = (idx, idy) if idx < idy else (idy, idx)
        if temp in self.edges:
            raise RuntimeError(f'edge:{edge} is already in pool')
        else:
            self.edges.add(temp)

    def search(self, node: int) -> Set[Tuple[int, int]]:
        """ Ищем ребра с концом равным node """
        temp = set()
        for edge in self.edges:
            if node == edge[0] or node == edge[1]:
                temp.add(edge)
        return temp

    def __contains__(self, item: Tuple[int, int]) -> bool:
        idx, idy = item
        temp = (idx, idy) if idx < idy else (idy, idx)
        return True if temp in self.edges else False


class Graph(PoolEdges):
    total_length: float = 0.0

    def add(self, edge: Tuple[int, int], price: float = 0) -> None:
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
