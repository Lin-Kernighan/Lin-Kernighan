from __future__ import annotations

from math import sqrt
from random import randint
from typing import List, Type, Tuple

from src.algorithms.heuristics.tsp_opt import TspOpt
from src.algorithms.heuristics.two_opt import TwoOpt
from src.structures.matrix import Matrix
from src.structures.tabu_list import TabuDict, AbstractTabu

Node = int


class TabuSearch:

    def __init__(self, tabu_list: AbstractTabu, tsp: Type[TspOpt], tour: List[Node], matrix: Matrix):
        self.data = tabu_list
        self.tour = tour
        self.matrix = matrix
        self.tsp = tsp

    @staticmethod
    def run(tour: List[Node], matrix: Matrix, depth: int) -> TabuSearch:
        search = TabuSearch(TabuDict(depth), TwoOpt, tour, matrix)
        search.optimize()
        return search

    def optimize(self, iteration=10) -> None:
        while iteration > 0:
            print(f'{iteration} : {self.best_result()[1]}')
            tsp = self.tsp(self.tour, self.matrix)
            self.tour = tsp.tabu_optimize(self.data)
            for _ in range(int(sqrt(len(self.tour)))):
                self.swap()
            iteration -= 1

    def swap(self) -> None:
        size = len(self.tour) - 1
        x = randint(0, size)
        while x == (y := randint(0, size)):
            continue
        self.tour[x], self.tour[y] = self.tour[y], self.tour[x]

    def best_tour(self) -> List[Node]:
        return self.data.best_tour()

    def best_result(self) -> Tuple[int, float]:
        return self.data.best_result()
