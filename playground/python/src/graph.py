from dataclasses import dataclass
from typing import List


@dataclass(order=True)
class Edge:
    price: float
    src: int
    dst: int

    def __str__(self) -> str:
        return f'{self.src}->{self.dst}'

    def __repr__(self) -> str:
        return str(self)


class Graph:
    edges: List[Edge]
    nodes: List[List[float]]

    def __init__(self, points: List[List[float]]) -> None:
        self.nodes = points
        self.edges = []
