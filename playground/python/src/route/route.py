from __future__ import annotations
from typing import List


class Node:
    index: int


class Route:

    def __init__(self, points: List[List[float]]):
        pass

    def predecessor(self, node: Node):
        pass

    def successor(self, node: Node):
        pass

    def between(self, forth: Node, back: Node) -> bool:
        pass

    def move(self) -> None:
        pass
