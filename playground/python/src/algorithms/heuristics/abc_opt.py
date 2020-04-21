from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.structures.collector import Collector
from src.structures.matrix import Matrix
from src.structures.tabu_list import AbstractTabu
from src.utils import get_length, rotate_zero

Node = int


class AbcOpt(ABC):

    def __init__(self, tour: List[Node], matrix: Matrix):
        self.tour = rotate_zero(tour)
        self.matrix = matrix
        self.size = len(tour)
        self.tabu_list: Optional[AbstractTabu] = None
        self.collector = None
        self.length = get_length(matrix, tour)

    @staticmethod
    def run(tour: List[Node], matrix: Matrix) -> AbcOpt:
        """ Полный запуск на точках """
        opt = AbcOpt(tour, matrix)
        opt.optimize()
        return opt

    @abstractmethod
    def optimize(self) -> List[Node]:
        """ Просто запуск эвристики """

    @abstractmethod
    def tabu_optimize(self, tabu_list: AbstractTabu, collector: Collector) -> List[Node]:
        """ Запуск эвристики под управление tabu search """
