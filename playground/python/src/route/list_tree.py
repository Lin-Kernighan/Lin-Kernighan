from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.route.route import Route


@dataclass
class Node:
    predecessor: Optional[Node]
    successor: Optional[Node]

    value: int
    reversed: bool


@dataclass
class Block:
    start: int
    end: int
    start_node: Optional[Node]  # end node?
    reversed: bool


class ArrayListTree(Route):
    blocks: List[Block]
    data: List[Node]

    data_length: int
    block_size: int
    blocks_length: int

    def __init__(self, points: List[Tuple[float, float]], order: List[int]) -> None:
        self.data_length = len(points)
        self.data = [Node(None, None, index, False) for index in range(self.data_length)]  # change value
        previous_node = self.data[order[-1]]
        for node in order:
            previous_node.successor = self.data[node]
            self.data[node].predecessor = previous_node
            previous_node = self.data[node]

        self.block_size = int(math.sqrt(self.data_length))  # sizeof Block
        self.blocks_length = self.block_size if self.block_size ** 2 == self.data_length else self.block_size + 1
        self.blocks = [Block(0, 0, None, False)] * self.blocks_length
        counter = 0
        for index in range(self.blocks_length):
            self.blocks[index] = Block(
                start=counter,
                start_node=self.data[counter],  # фигово, что counter тут рассчитываю
                end=self.data_length if (counter := counter + self.block_size) > self.data_length else counter,
                reversed=False)

    def __str__(self) -> str:
        return f'blocks({len(self.blocks)}):\n{self.blocks}\n'

    def __repr__(self) -> str:
        return str(self)

    def len_data(self) -> int:
        return self.data_length

    def len_blocks(self) -> int:
        return self.blocks_length

    def predecessor(self, node: Node) -> Optional[Node]:
        return node.predecessor

    def successor(self, node: Node) -> Optional[Node]:
        return node.successor

    def between(self, forth: Node, back: Node, search: Node) -> bool:
        found = False  # по ходу движения проверок нет?
        while forth is not None and back is not None:
            if forth == search or back == search:
                found = True
            if forth == back or forth.successor == back:
                break
            forth = forth.successor
            back = back.predecessor

        return found

    def node(self, node: int) -> Optional[Node]:
        """ Get by index in weight matrix or order in tsp file """
        if 0 <= node < self.len_data():
            return self.data[node]
        return None

    def __getitem__(self, index: int) -> Optional[Node]:
        """ Get by order in tsp tour """
        if 0 <= index < self.len_data():
            block_number = index // self.block_size
            node_number = index % self.block_size
            block = self.blocks[block_number]

            counter = 0
            node = block.start_node
            while counter != node_number:
                node = node.successor
                counter += 1
            return node
        return None
