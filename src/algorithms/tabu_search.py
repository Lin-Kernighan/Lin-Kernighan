import logging
from random import randint
from sys import maxsize
from typing import Tuple

import numpy as np

from src.algorithms.structures.collector import Collector
from src.algorithms.structures.tabu_list import TabuSet


class TabuSearch:

    def __init__(self, opt, length: float, tour: np.ndarray, matrix: np.ndarray):
        self.collector = Collector(['length', 'gain'], {'tabu search': len(tour)})
        self.length, self.tour, self.matrix = length, tour, matrix
        self.data = TabuSet()
        self.opt = opt

    def optimize(self, iteration=10, swap=2) -> None:
        """ Прогон """
        self.collector.update({'length': self.length, 'gain': 0})
        best_cost = maxsize
        while iteration > 0:
            self.opt.length, self.opt.tour = self.length, self.tour.copy()
            self.opt.tabu_optimize(self.data, self.collector)
            if best_cost > self.best_tour()[0]:
                self.length, self.tour = self.best_tour()
            logging.info(f'{iteration} : {self.best_tour()[0]} : {self.length}')
            for _ in range(swap):
                self.swap()
            iteration -= 1
        self.length, self.tour = self.best_tour()

    def swap(self) -> None:
        """ Попытка сломать тур """
        size = len(self.tour) - 1
        x = randint(0, size)
        while x == (y := randint(0, size)):
            continue
        self.tour[x], self.tour[y] = self.tour[y], self.tour[x]

    def best_tour(self) -> Tuple[float, np.ndarray]:
        """ Лучший тур """
        return self.data.best_result(), self.data.best_tour()
