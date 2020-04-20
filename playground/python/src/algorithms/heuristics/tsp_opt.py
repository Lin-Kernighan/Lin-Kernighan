from __future__ import annotations

from typing import List, Optional

from src.structures.collector import Collector
from src.structures.matrix import Matrix
from src.structures.tabu_list import AbstractTabu
from src.utils import get_length, rotate_zero

Node = int


class TspOpt:

    def __init__(self, tour: List[Node], matrix: Matrix):
        self.tour = rotate_zero(tour)
        self.matrix = matrix
        self.size = len(tour)
        self.tabu_list: Optional[AbstractTabu] = None
        self.length = get_length(matrix, tour)

    @staticmethod
    def run(tour: List[Node], matrix: Matrix) -> TspOpt:
        """ Полный запуск на точках """
        opt = TspOpt(tour, matrix)
        opt.optimize()
        return opt

    def optimize(self) -> List[Node]:
        pass

    def tabu_optimize(self, tabu_list: AbstractTabu, collector: Collector) -> List[Node]:
        pass
