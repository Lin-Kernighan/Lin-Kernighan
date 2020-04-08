from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from blist import *

from src.route.route import Route
from src.weight_matrix import WeightMatrix


@dataclass
class Node:
    value: int


class RouteBlist(Route):
    data: blist[Node]
    weight_matrix: WeightMatrix

    def __init__(self, points: List[List[float]]) -> None:
        self.weight_matrix = WeightMatrix(points)
        self.data = blist([Node(0)])  # wtf
        self.data *= len(points)
        for index, _ in enumerate(points):  # some initial tour generator
            self.data[index] = index

    def predecessor(self, node: Node) -> Optional[Node]:  # what is better index of Node or Node?
        index = self.data.index(node)
        if 0 < index < len(self.data):
            return self.data[index - 1]
        return None

    def successor(self, node: Node) -> Optional[Node]:  # what is better index of Node or Node?
        index = self.data.index(node)
        if 0 <= index < len(self.data) - 1:
            return self.data[index + 1]
        return None

    def between(self, forth: Node, back: Node, search: Node) -> bool:
        index_forth = self.data.index(forth)
        index_back = self.data.index(back)
        index_search = self.data.index(search)

        if index_forth < index_back < index_search:
            return True
        return False

    def move(self) -> None:
        pass
