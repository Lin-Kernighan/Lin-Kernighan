from dataclasses import dataclass
from typing import Set


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
class Graph:
    edges: Set[Edge]  # набор ребер
