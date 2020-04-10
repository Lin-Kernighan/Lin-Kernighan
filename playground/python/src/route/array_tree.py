import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.route.route import Route


@dataclass
class Node:
    index: int
    value: int
    reversed: bool


@dataclass
class Block:
    start: int
    end: int
    reversed: bool


class ArrayTree(Route):
    blocks: List[Block]
    data: List[Node]

    def __init__(self, points: List[Tuple[float, float]]) -> None:
        length = len(points)

        self.data = [Node(0, 0, False)] * length
        for index, _ in enumerate(points):  # some initial tour generator
            self.data[index] = Node(index, index, False)

        size = int(math.sqrt(length))  # sizeof Block
        blocks_length = size if size ** 2 == length else size + 1
        self.blocks = [Block(0, 0, False)] * blocks_length
        tmp = 0
        for index in range(blocks_length):
            self.blocks[index] = Block(tmp, length if (tmp := tmp + size) > length else tmp, False)

    def __str__(self) -> str:
        return f'blocks({len(self.blocks)}):\n{self.blocks}\ndata({len(self.data)}):\n{self.data}'

    def __repr__(self) -> str:
        return str(self)

    def len_data(self) -> int:
        return len(self.data)

    def len_blocks(self) -> int:
        return len(self.blocks)

    def predecessor(self, node: Node) -> Optional[Node]:
        if 0 < node.index < self.len_data():
            return self.data[node.index - 1]
        return None

    def successor(self, node: Node) -> Optional[Node]:
        if 0 <= node.index < self.len_data() - 1:
            return self.data[node.index + 1]
        return None

    def between(self, forth: Node, back: Node, search: Node) -> bool:
        if forth.index < search.index < back.index:
            return True
        return False
