from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.structures.tour.abc_tour import AbcTour

Point = Tuple[float, float]
Edge = Tuple[int, int]


# TODO: перевороты


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
class TreeTour(AbcTour):

    def __init__(self, points: List[Point], order: List[int]):
        self.size = len(points)
        self.data = [Node(None, None, index) for index in range(self.size)]  # change value
        self.block_size = int(math.sqrt(self.size))  # sizeof Block
        self.blocks_length = self.block_size if self.block_size ** 2 == self.size else self.block_size + 1
        self.blocks = [Block(0, 0, None)] * self.blocks_length

        previous_node = self.data[order[-1]]
        for node in order:
            previous_node.successor = self.data[node]
            self.data[node].predecessor = previous_node
            previous_node = self.data[node]

        counter = 0
        for index in range(self.blocks_length):
            self.blocks[index] = Block(
                start=counter,
                start_node=self.data[counter],  # фигово, что counter тут рассчитываю
                end=self.size if (counter := counter + self.block_size) > self.size else counter,
                reversed=False)

    def __len__(self):
        """ Кол-во вершин """
        return self.size

    def __getitem__(self, index: int) -> Node:
        """ По индексу в туре """
        index %= self.size
        block_number = index // self.block_size
        node_number = index % self.block_size
        block = self.blocks[block_number]

        counter = 0
        node = block.start_node
        while counter != node_number:
            node = node.successor
            counter += 1
        return node

    def __contains__(self, edge: Edge) -> bool:
        """ Наличие ребра в туре """
        pass

    def index(self, node: Node) -> int:
        """ Номер вершины в туре """
        return self.data.index(node)

    def around(self, node) -> Tuple[Node, Node]:
        """ Предыдущая вершина и следующая текущей веришны """
        return node.predecessor, node.predecessor

    def predecessor(self, node: Node) -> Node:
        """ Предыдущий """
        return node.predecessor

    def successor(self, node: Node) -> Node:
        """ Следующий """
        return node.successor

    def between(self, start: Node, end: Node, search: Node) -> bool:
        """  Находится ли вершина search между вершиной start и вершиной end """
        found = False  # TODO: проверка на тур норм и его не надо reverse
        while start.successor != end:  # TODO: проверка на зацикливание?
            if start == search:
                found = True
            start = start.successor
        return found

    def head(self) -> Node:
        return self.data[0]

    def edges(self) -> List[Edge]:
        """ Возвращает как список ребер """
        head = self.head()
        runner = head.successor
        temp = []
        while runner.value != head.value:
            temp.append((runner.predecessor.value, runner.value))
            runner = runner.successor
        temp.append((runner.predecessor.value, runner.value))
        return temp

    def nodes(self) -> List[int]:
        """ Список вершин в правильном порядке """
        temp = [0] * self.size
        i = 0
        head = runner = self.head()
        while runner.successor is not head:
            temp[i] = runner.value
            runner = runner.successor
            i += 1
        return temp

    def generate(self, broken: set, joined: set) -> Tuple[bool, list]:
        """ Создаем новый тур, а потом проверяем его на целостность и наличие циклов
        broken: удаляемые ребра
        joined: добавляемые ребра
        """
        # TODO: this shit
        pass

    def reverse(self, start, end) -> None:
        """ Переворот куска тура """
        # TODO: this shit
        pass
