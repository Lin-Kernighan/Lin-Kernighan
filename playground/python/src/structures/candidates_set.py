from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.structures.matrix import Matrix


@dataclass(order=True)
class Candidate:
    price: float
    in_work: bool
    src: int
    dst: int

    @staticmethod
    def build(x: int, y: int, price: float) -> Candidate:
        return Candidate(price, True, x, y) if x < y else Candidate(price, True, y, x)


@dataclass
class CandidatesSet:
    data: Dict[int, List[Candidate]] = field(default_factory=dict)  # все отсортированные ребра
    iterators: List[int] = field(default_factory=list)  #

    @staticmethod
    def build(matrix: Matrix, excess: float) -> CandidatesSet:
        candidate_set = CandidatesSet()
        for idx in range(0, matrix.dimension):
            for idy in range(idx + 1, matrix.dimension):
                if matrix[idx][idy] > excess:
                    candidate_set.add_link(idx, idy, matrix[idx][idy])
        for key in candidate_set.data:
            sorted(candidate_set.data[key])
        candidate_set.iterators = [0] * matrix.dimension
        return candidate_set

    def add_link(self, x: int, y: int, price: float) -> None:
        """ Add link to data x<->y """
        edge = Candidate.build(x, y, price)
        self.__add_edge(x, edge)
        self.__add_edge(y, edge)

    def get_candidate(self, x: int) -> Optional[Candidate]:
        length = len(self.data[x])
        current = self.iterators[x]
        while current < length:
            candidate = self.data[x][current]
            if not candidate.in_work:
                self.iterators[x] = current
                candidate.in_work = False
                return candidate
            else:
                current += 1
        return None

    def __add_edge(self, x: int, edge: Candidate) -> None:
        """ Add edge for x node """
        if x in self.data:
            self.data[x].append(edge)
        else:
            self.data.update({x: [edge]})
