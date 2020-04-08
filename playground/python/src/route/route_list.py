from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.route.route import Route
from src.weight_matrix import WeightMatrix


@dataclass
class Node:
    index: int
    predecessor: Node
    successor: Node

    def __init__(self, index: int, predecessor: Optional[Node], successor: Optional[Node]) -> None:
        self.index = index
        self.predecessor = predecessor
        self.successor = successor


class RouteList(Route):
    head: Node
    weight_matrix: WeightMatrix

    def __init__(self, points: List[List[float]]) -> None:
        self.weight_matrix = WeightMatrix(points)
        prev: Optional[Node] = None
        for index, _ in enumerate(points):  # some initial tour generator
            if prev is not None:
                new_node = Node(index, prev, None)
                prev.successor = new_node
                prev = new_node
            else:
                self.head = Node(index, None, None)
                prev = self.head

    def predecessor(self, node: Node) -> Node:
        return node.predecessor

    def successor(self, node: Node) -> Node:
        return node.successor

    def between(self, forth: Node, back: Node) -> bool:
        while forth is not None and back is not None:
            if forth == back or forth.successor == back:
                return True
            forth = forth.successor
            back = back.predecessor
        return False

    def move(self) -> None:
        pass
