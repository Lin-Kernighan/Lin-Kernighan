from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.structures.graph import Edge


@dataclass
class Node:
    predecessor: Optional[Node]
    successor: Optional[Node]
    value: int
    reversed: bool = False


@dataclass
class Block:
    start: int
    end: int
    start_node: Optional[Node]  # end node?
    reversed: bool = False


@dataclass
class Route:
    data_length: int

    blocks: List[Block] = field(init=False)
    data: List[Node] = field(init=False)
    block_size: int = field(init=False)
    blocks_length: int = field(init=False)

    def __post_init__(self):
        self.data = [Node(None, None, index) for index in range(self.data_length)]  # change value
        self.block_size = int(math.sqrt(self.data_length))  # sizeof Block
        self.blocks_length = self.block_size if self.block_size ** 2 == self.data_length else self.block_size + 1
        self.blocks = [Block(0, 0, None)] * self.blocks_length

    @staticmethod
    def build(points: List[Tuple[float, float]], order: List[int]) -> Route:
        route = Route(len(points))
        previous_node = route.data[order[-1]]
        for node in order:
            previous_node.successor = route.data[node]
            route.data[node].predecessor = previous_node
            previous_node = route.data[node]

        counter = 0
        for index in range(route.blocks_length):
            route.blocks[index] = Block(
                start=counter,
                start_node=route.data[counter],  # фигово, что counter тут рассчитываю
                end=route.data_length if (counter := counter + route.block_size) > route.data_length else counter,
                reversed=False)
        return route

    @staticmethod
    def predecessor(node: Node) -> Optional[Node]:
        return node.predecessor

    @staticmethod
    def successor(node: Node) -> Optional[Node]:
        return node.successor

    @staticmethod
    def between(forth: Node, back: Node, search: Node) -> bool:
        found = False  # по ходу движения проверок нет?
        while forth is not None and back is not None:
            if forth == search or back == search:
                found = True
            if forth == back or forth.successor == back:
                break
            forth = forth.successor
            back = back.predecessor

        return found

    def edges(self) -> List[Edge]:
        head = self.data[0]
        runner = head.successor
        temp = []
        while runner.value != head.value:
            temp.append(Edge(0, runner.predecessor.value, runner.value))
            runner = runner.successor
        temp.append(Edge(0, runner.predecessor.value, runner.value))
        return temp

    def node(self, node: int) -> Optional[Node]:
        """ Get by index in weight matrix or order in tsp file """
        if 0 <= node < self.len_data():
            return self.data[node]
        return None

    def len_data(self) -> int:
        return self.data_length

    def len_blocks(self) -> int:
        return self.blocks_length

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

    def __str__(self) -> str:
        return f'blocks({len(self.blocks)}):\n{self.blocks}\n'

    def __repr__(self) -> str:
        return str(self)
