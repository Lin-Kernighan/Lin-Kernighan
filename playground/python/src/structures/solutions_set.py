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

    def add(self, graph: Graph) -> None:
        self.solutions.append(graph)
        if graph.total_length < self.solutions[self.minimum].total_length:
            self.minimum = len(self.solutions)
