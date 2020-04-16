from collections import OrderedDict
from dataclasses import dataclass
from sys import maxsize
from typing import List, Tuple

from src.utils import rotate_zero

Node = int


class AbstractTabu:
    def contains(self, item: List[Node]) -> bool:
        pass

    def append(self, tour: List[Node], length: float) -> bool:
        pass

    def best_result(self) -> Tuple[int, float]:
        pass

    def best_tour(self) -> List[Node]:
        pass


@dataclass
class TabuDict(AbstractTabu):
    size: int
    data = OrderedDict()
    index = 0
    best_length, this_index, best_route = maxsize, 0, None

    def contains(self, item: List[Node]) -> bool:
        if str(rotate_zero(item)) in self.data:
            return True
        return False

    def append(self, tour: List[Node], length: float) -> bool:
        tour = rotate_zero(tour)
        if self.contains(tour):
            return False
        if self.size != -1 and len(self.data) == self.size:
            self.data.popitem(last=False)
        self.data[str(tour)] = (self.index, length)
        if length < self.best_length:
            self.best_route, self.this_index, self.best_length = tour, self.index, length
        self.index += 1
        return True

    def best_result(self) -> Tuple[int, float]:
        return self.this_index, self.best_length

    def best_tour(self) -> List[Node]:
        return self.best_route


@dataclass
class TabuFilterBloom:
    # TODO: with filter bloom
    pass
