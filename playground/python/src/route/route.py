from __future__ import annotations
from typing import Optional


class Node:
    value: int


class Route:

    def predecessor(self, node: Node) -> Optional[Node]:
        pass

    def successor(self, node: Node) -> Optional[Node]:
        pass

    def between(self, forth: Node, back: Node, search: Node) -> bool:
        pass

    def move(self) -> None:
        pass
