from __future__ import annotations

from typing import Tuple

# import numba as nb
import numpy as np

from src.algorithms.heuristics.abc_opt import AbcOpt
from src.structures.collector import Collector
from src.structures.tabu_list import TabuSet
from src.utils import right_rotate


# @nb.njit
def _swap(tour: np.ndarray, x: int, y: int) -> np.ndarray:
    """ Меняем местами два элемента и разворачивает все что между ними """
    size, temp = len(tour), 0
    if x < y:
        temp = (y - x + 1) // 2
    elif x > y:
        temp = ((size - x) + y + 2) // 2
    for i in range(temp):
        first, second = (x + i) % size, (y - i) % size
        tour[first], tour[second] = tour[second], tour[first]
    return tour


# @nb.njit
def _improve(matrix: np.ndarray, tour: np.ndarray) -> Tuple[tuple, float]:
    """ Просто пробег по вершинам, ищем лучшее """
    best_change, saved = 0, None
    size = matrix.shape[0]

    for n in range(matrix.shape[0]):
        for m in range(n + 2, matrix.shape[0]):
            i, j = tour[n % size], tour[m % size]
            x, y = tour[(n + 1) % size], tour[(m + 1) % size]
            change = matrix[i][j] + matrix[x][y]
            change -= matrix[i][x] + matrix[j][y]
            if change < best_change:
                best_change = change
                saved = (n, m)

    return saved, best_change


class TwoOpt(AbcOpt):

    def __init__(self, length: float, tour: np.ndarray, matrix: np.ndarray):
        super().__init__(length, tour, matrix)

    def optimize(self) -> np.ndarray:
        """ Запуск """
        best_change, iteration, self.collector = -1, 0, Collector(['length', 'gain'], {'two_opt': self.size})
        self.collector.update({'length': self.length, 'gain': 0})
        print(f'start : {self.length}')

        while best_change < 0:
            best_change = self.__two_opt()
            if best_change >= 0:
                self.tour = right_rotate(self.tour, len(self.tour) // 3)
                best_change = self.__two_opt()
            self.length += best_change
            self.collector.update({'length': self.length, 'gain': -best_change})
            print(f'{iteration} : {self.length}')
            iteration += 1

        return self.tour

    def tabu_optimize(self, tabu_list: TabuSet, collector: Collector) -> np.ndarray:
        """ 2-opt для Tabu search """
        self.tabu_list, best_change, self.collector = tabu_list, -1, collector
        self.collector.update({'length': self.length, 'gain': 0})

        while best_change < 0:
            best_change = self.__tabu_two_opt()
            self.length += best_change
            tabu_list.append(self.tour, self.length)
            self.collector.update({'length': self.length, 'gain': -best_change})

        return self.tour

    def __two_opt(self) -> float:
        """ Просто 2-opt """
        saved, best_change = _improve(self.matrix, self.tour)
        if best_change < 0:
            i, j = saved
            self.tour = _swap(self.tour, i + 1, j)
        return best_change

    def __tabu_two_opt(self) -> float:
        """ 2-opt и проверка """
        saved, best_change = _improve(self.matrix, self.tour)  # улучшили

        if best_change < 0:
            i, j = saved
            tour = _swap(self.tour, i + 1, j)
            if self.tabu_list.contains(tour):
                return 0.0
            else:
                self.tour = tour  # если не в табу, сохранили

        return best_change
