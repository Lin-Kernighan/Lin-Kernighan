from __future__ import annotations

from random import randint
from sys import maxsize
from typing import List, Type, Tuple

from src.algorithms.heuristics.abc_opt import AbcOpt
from src.structures.collector import Collector
from src.structures.matrix import Matrix
from src.structures.tabu_list import TabuSet, AbstractTabu
from src.utils import get_length

Node = int


class TabuSearch:

    def __init__(self, tabu_list: AbstractTabu, tsp: Type[AbcOpt], tour: List[Node], matrix: Matrix):
        self.collector = Collector(['length', 'gain'], {'tabu search': len(tour), 'type': tsp.__name__})
        self.data = tabu_list
        self.tour = tour
        self.matrix = matrix
        self.tsp = tsp
        self.length = get_length(self.matrix, tour)

    @staticmethod
    def run(tour: List[Node], matrix: Matrix, opt: Type[AbcOpt], depth: int) -> TabuSearch:
        """ Полный цикл работы за вас """
        search = TabuSearch(TabuSet(depth), opt, tour, matrix)
        search.optimize()
        return search

    def optimize(self, iteration=10, swap=2) -> None:
        """ Прогон """
        self.collector.update({'length': self.length, 'gain': 0})
        best_cost = maxsize
        while iteration > 0:
            tsp = self.tsp(self.tour, self.matrix)
            tsp.tabu_optimize(self.data, self.collector)
            if best_cost > self.best_result()[1]:
                self.tour = self.best_tour()
            print(f'{iteration} : {self.best_result()[1]} : {tsp.length}')
            for _ in range(swap):
                self.swap()
            iteration -= 1
        self.tour = self.best_tour()

    def swap(self) -> None:
        """ Попытка сломать тур """
        size = len(self.tour) - 1
        x = randint(0, size)
        while x == (y := randint(0, size)):
            continue
        self.tour[x], self.tour[y] = self.tour[y], self.tour[x]

    def best_tour(self) -> List[Node]:
        """ Лучший тур """
        return self.data.best_tour()

    def best_result(self) -> Tuple[int, float]:
        """ Когда был добавлен лучший тур и его длина """
        return self.data.best_result()
