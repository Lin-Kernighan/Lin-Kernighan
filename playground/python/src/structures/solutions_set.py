from typing import List, Optional

from src.structures.graph import Graph


class SolutionSet:
    solutions: List[Graph]  # список всех (вероятно не всех, но это потом) полученных решений
    minimum: int  # самое оптимальное в нем

    def __init__(self) -> None:
        self.solutions = []
        self.minimum = -1

    def get_best(self) -> Optional[Graph]:
        if self.minimum == -1:
            return None
        return self.solutions[self.minimum]
