from __future__ import annotations

from typing import List, Tuple

from src.algorithms.heuristics.tsp_opt import TspOpt
from src.structures.matrix import Matrix
from src.structures.tabu_list import AbstractTabu
from src.utils import right_rotate

Node = int


class TwoOpt(TspOpt):

    def __init__(self, tour: List[Node], matrix: Matrix):
        super().__init__(tour, matrix)

    def optimize(self) -> List[int]:
        """ Запуск """
        best_change, iteration = -1, 0
        print(f'start : {self.length}')

        while best_change < 0:
            best_change = self.__two_opt()
            if best_change >= 0:
                self.tour = right_rotate(self.tour, len(self.tour) // 3)  # костыль
                best_change = self.__two_opt()
            self.length += best_change
            print(f'{iteration} : {self.length}')  # оставлю, чтобы видеть, что алгоритм не помер еще
            iteration += 1

        return self.tour

    def tabu_optimize(self, tabu_list: AbstractTabu) -> List[Node]:
        """ 2-opt для Tabu search """
        self.tabu_list, best_change, iteration = tabu_list, -1, 0
        while best_change < 0:
            best_change = self.__tabu_two_opt()
            if best_change >= 0:  # да, все тот же костыль
                rotate = len(self.tour) // 3 * (iteration % 2 + 1)
                best_change = self.__tabu_two_opt(rotate)
            self.length += best_change
            iteration += 1
            tabu_list.append(self.tour, self.length)

        return self.tour

    def __two_opt(self) -> float:
        """ Просто 2-opt """
        saved, best_change = self.__improve(self.tour)
        if best_change < 0:
            i, j = saved
            self.tour = self.__swap(self.tour, i + 1, j)
        return best_change

    def __tabu_two_opt(self, rotate=0) -> float:
        """ 2-opt и проверка """
        tour = self.tour if rotate != 0 else right_rotate(self.tour, rotate)  # прокрутили
        saved, best_change = self.__improve(tour)  # улучшили

        if best_change < 0:
            i, j = saved
            tour = self.__swap(tour, i + 1, j)  # если норм, свапнули
            tour = tour if rotate != 0 else right_rotate(tour, -rotate)  # вернули обратно
            if self.tabu_list.contains(tour):
                return 0.0
            else:
                self.tour = tour  # если не в табу, сохранили

        return best_change

    def __improve(self, tour: List[int]) -> Tuple[tuple, float]:
        """ Просто пробег по вершинам, ищем лучшее """
        best_change, saved = 0, None

        for n in range(self.matrix.dimension - 3):
            for m in range(n + 2, self.matrix.dimension - 1):
                i, j = tour[n], tour[m]
                x, y = tour[n + 1], tour[m + 1]
                change = self.matrix[i][j] + self.matrix[x][y]
                change -= self.matrix[i][x] + self.matrix[j][y]
                if change < best_change:
                    best_change = change
                    saved = (n, m)

        return saved, best_change

    @staticmethod
    def __swap(tour: List[int], i: int, j: int) -> List[int]:
        """ Меняем местами два элемента и разворачивает все что между ними """
        return tour[:i] + list(reversed(tour[i:j + 1])) + tour[j + 1:]
