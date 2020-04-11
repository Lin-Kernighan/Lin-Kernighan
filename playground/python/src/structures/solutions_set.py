from dataclasses import dataclass, field
from typing import List, Optional

from src.structures.graph import Graph


@dataclass
class SolutionSet:
    solutions: List[Graph] = field(default_factory=list)  # список всех полученных решений
    minimum: int = -1  # самое оптимальное в нем

    def get_best(self) -> Optional[Graph]:
        if self.minimum == -1:
            return None
        return self.solutions[self.minimum]

    def add(self, graph: Graph) -> None:
        self.solutions.append(graph)
        if graph.total_length < self.solutions[self.minimum].total_length:
            self.minimum = len(self.solutions)
