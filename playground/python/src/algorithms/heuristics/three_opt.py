from __future__ import annotations

from typing import List, Tuple

from src.algorithms.heuristics.abc_opt import AbcOpt
from src.structures.collector import Collector
from src.structures.matrix import Matrix
from src.structures.tabu_list import AbstractTabu
from src.utils import right_rotate

Point = Tuple[float, float]
Node = int


# TODO: починить swap, чтобы не дергать rotate


class ThreeOpt(AbcOpt):

    def __init__(self, tour: List[Node], matrix: Matrix):
        super().__init__(tour, matrix)
        self.collector = None

    def optimize(self) -> List[int]:
        """ Запуск """
        best_gain, iteration, self.collector = 1, 0, Collector(['length', 'gain'], {'three_opt': self.size})
        self.collector.update({'length': self.length, 'gain': 0})
        print(f'start : {self.length}')

        while best_gain > 0:
            best_gain = self.__three_opt()
            if best_gain <= 0:
                self.tour = right_rotate(self.tour, len(self.tour) // 3)
                best_gain = self.__three_opt()
            self.length -= best_gain
            self.collector.update({'length': self.length, 'gain': best_gain})
            print(f'{iteration} : {self.length}')
            iteration += 1

        return self.tour

    def tabu_optimize(self, tabu_list: AbstractTabu, collector: Collector) -> List[Node]:
        """ 3-opt для Tabu search """
        self.tabu_list, best_gain, iteration, self.collector = tabu_list, 1, 0, collector
        self.collector.update({'length': self.length, 'gain': 0})

        while best_gain > 0:
            best_gain = self.__tabu_three_opt()
            if best_gain <= 0:
                rotate = len(self.tour) // 3 * (iteration % 2 + 1)
                best_gain = self.__tabu_three_opt(rotate)
            self.length -= best_gain
            self.collector.update({'length': self.length, 'gain': best_gain})
            tabu_list.append(self.tour, self.length)
            iteration += 1

        return self.tour

    def __three_opt(self) -> float:
        """ 3-opt """
        best_exchange, best_gain, best_nodes = self.__improve(self.tour)
        if best_gain > 0:
            self.tour = ThreeOpt.__exchange(self.tour, best_exchange, best_nodes)
        return best_gain

    def __tabu_three_opt(self, rotate=0) -> float:
        tour = self.tour if rotate != 0 else right_rotate(self.tour, rotate)
        best_exchange, best_gain, best_nodes = self.__improve(tour)
        if best_gain > 0:
            tour = ThreeOpt.__exchange(tour, best_exchange, best_nodes)
            tour = tour if rotate != 0 else right_rotate(tour, -rotate)
            if self.tabu_list.contains(tour):
                return 0.0
            else:
                self.tour = tour
        return best_gain

    def __improve(self, tour: List[int]) -> Tuple[int, float, tuple]:
        """ 3-opt пробег по вершинам """
        best_exchange, best_gain, best_nodes = 0, 0, None
        size = self.matrix.dimension

        for x in range(size - 5):
            for y in range(x + 2, size - 3):
                for z in range(y + 2, size - 1):
                    exchange, gain = ThreeOpt.__search(self.matrix, tour, x, y, z)
                    if gain > best_gain:
                        best_gain, best_exchange, best_nodes = gain, exchange, (x, y, z)

        return best_exchange, best_gain, best_nodes

    @staticmethod
    def __search(matrix: Matrix, tour: List[int], x: int, y: int, z: int) -> Tuple[int, float]:
        """ Поиск лучшего среди переборов """
        a, b, c, d, e, f = tour[x], tour[x + 1], tour[y], tour[y + 1], tour[z], tour[z + 1]
        base = current_min = matrix[a][b] + matrix[c][d] + matrix[e][f]
        gain = 0
        exchange = -1

        if current_min > (current := matrix[a][e] + matrix[c][d] + matrix[b][f]):  # 2-opt (a, e) [d, c] (b, f)
            gain, exchange, current_min = base - current, 0, current
        if current_min > (current := matrix[a][b] + matrix[c][e] + matrix[d][f]):  # 2-opt [a, b] (c, e) (d, f)
            gain, exchange, current_min = base - current, 1, current
        if current_min > (current := matrix[a][c] + matrix[b][d] + matrix[e][f]):  # 2-opt (a, c) (b, d) [e, f]
            gain, exchange, current_min = base - current, 2, current
        if current_min > (current := matrix[a][d] + matrix[e][c] + matrix[b][f]):  # 3-opt (a, d) (e, c) (b, f)
            gain, exchange, current_min = base - current, 3, current
        if current_min > (current := matrix[a][d] + matrix[e][b] + matrix[c][f]):  # 3-opt (a, d) (e, b) (c, f)
            gain, exchange, current_min = base - current, 4, current
        if current_min > (current := matrix[a][e] + matrix[d][b] + matrix[c][f]):  # 3-opt (a, e) (d, b) (c, f)
            gain, exchange, current_min = base - current, 5, current
        if current_min > (current := matrix[a][c] + matrix[b][e] + matrix[d][f]):  # 3-opt (a, c) (b, e) (d, f)
            gain, exchange, current_min = base - current, 6, current

        return exchange, gain

    @staticmethod
    def __exchange(tour: List[int], best_exchange: int, nodes: tuple) -> List[int]:
        """ Конечная замена """
        x, y, z = nodes
        a, b, c, d, e, f = x, x + 1, y, y + 1, z, z + 1
        sol = []
        if best_exchange == 0:
            sol = tour[:a + 1] + tour[e:d - 1:-1] + tour[c:b - 1:-1] + tour[f:]
        elif best_exchange == 1:
            sol = tour[:a + 1] + tour[b:c + 1] + tour[e:d - 1:-1] + tour[f:]
        elif best_exchange == 2:
            sol = tour[:a + 1] + tour[c:b - 1:-1] + tour[d:e + 1] + tour[f:]
        elif best_exchange == 3:
            sol = tour[:a + 1] + tour[d:e + 1] + tour[c:b - 1:-1] + tour[f:]
        elif best_exchange == 4:
            sol = tour[:a + 1] + tour[d:e + 1] + tour[b:c + 1] + tour[f:]
        elif best_exchange == 5:
            sol = tour[:a + 1] + tour[e:d - 1:-1] + tour[b:c + 1] + tour[f:]
        elif best_exchange == 6:
            sol = tour[:a + 1] + tour[c:b - 1:-1] + tour[e:d - 1:-1] + tour[f:]
        return sol
