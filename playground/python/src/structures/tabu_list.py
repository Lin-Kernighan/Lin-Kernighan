from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from sys import maxsize
from typing import List, Tuple

from orderedset import OrderedSet

from src.utils import rotate_zero

Node = int


class AbstractTabu(ABC):

    @abstractmethod
    def contains(self, item: List[Node]) -> bool:
        pass

    @abstractmethod
    def append(self, tour: List[Node], length: float) -> bool:
        pass

    @abstractmethod
    def best_result(self) -> Tuple[int, float]:
        pass

    @abstractmethod
    def best_tour(self) -> List[Node]:
        pass


@dataclass
class TabuSet(AbstractTabu):
    size: int
    data = OrderedSet()
    index = 0
    best_length, this_index, best_route = maxsize, 0, None

    def contains(self, item: List[Node]) -> bool:
        if str(rotate_zero(item)) in self.data:
            return True
        return False

    def __add(self, item: List[Node]) -> None:
        self.data.add(str(item))

    def append(self, tour: List[Node], length: float) -> bool:
        tour = rotate_zero(tour)
        if self.contains(tour):
            return False
        if self.size != -1 and len(self.data) == self.size:
            self.data.pop(last=False)
        self.__add(tour)
        if length < self.best_length:
            self.best_route, self.this_index, self.best_length = deepcopy(tour), self.index, length
        self.index += 1
        return True

    def best_result(self) -> Tuple[int, float]:
        return self.this_index, self.best_length

    def best_tour(self) -> List[Node]:
        return deepcopy(self.best_route)


@dataclass
class TabuFilterBloom:
    # TODO: with filter bloom
    pass
