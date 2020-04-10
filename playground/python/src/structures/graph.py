from dataclasses import dataclass
from typing import Set, Tuple


@dataclass(order=True)
class Edge:
    price: float
    src: int
    dst: int

    def __str__(self) -> str:
        return f'{self.src}->{self.dst}'

    def __repr__(self) -> str:
        return str(self)


class PoolEdges:
    edges: Set[Tuple[int, int]]

    def __init__(self) -> None:
        self.edges = set()

    def add(self, edge: Tuple[int, int]) -> None:
        idx, idy = edge
        if idx == idy:
            raise RuntimeError(f'idx == idy in edge:{edge}')

        temp = (idx, idy) if idx < idy else (idy, idx)
        if temp in self.edges:
            raise RuntimeError(f'edge:{edge} is already in pool')
        else:
            self.edges.add(temp)

    def search(self, node: int) -> Set[Tuple[int, int]]:
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
    total_length: float

    def __init__(self) -> None:
        super().__init__()
        self.total_length = 0

    def add(self, edge: Tuple[int, int], price: float = 0) -> None:
        idx, idy = edge
        if idx == idy:
            raise RuntimeError(f'idx == idy in edge:{edge}')

        temp = (idx, idy) if idx < idy else (idy, idx)
        if edge in self.edges:
            raise RuntimeError(f'edge:{edge} is already in pool')
        else:
            self.edges.add(edge)
            self.total_length += price
