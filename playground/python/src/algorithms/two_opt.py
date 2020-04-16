from typing import List, Tuple

from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import Matrix
from src.utils import right_rotate, get_length, Point


class TwoOpt:

    @staticmethod
    def run(points: List[Point]) -> List[int]:
        """ Полный запуск на точках """
        matrix = Matrix.weight_matrix(points)
        init_tour = InitialTour.greedy(matrix)
        return TwoOpt.optimize(init_tour, matrix)

    @staticmethod
    def optimize(tour: List[int], matrix: Matrix) -> List[int]:
        """ Запуск на готовом туре и матрице смежностей """
        best_change = -1
        iteration = 0
        length = get_length(matrix, tour)
        print(f'start : {length}')
        while best_change < 0:
            best_change, tour = TwoOpt.__two_opt(matrix, tour)
            if best_change == 0:
                tour = right_rotate(tour, len(tour) // 3)  # костылек, зато пока работает быстрее чем алгоритм, забью
                best_change, tour = TwoOpt.__two_opt(matrix, tour)
            length += best_change
            print(f'{iteration} : {length}')  # оставлю, чтобы видеть, что алгоритм не помер еще
            iteration += 1
        return tour

    @staticmethod
    def __two_opt(matrix: Matrix, tour: List[int]) -> Tuple[float, List[int]]:
        """ 2-opt """
        saved, best_change = TwoOpt.__improve(matrix, tour)
        if best_change < 0:
            i, j = saved
            tour = TwoOpt.__swap(tour, i + 1, j)
        return best_change, tour

    @staticmethod
    def __improve(matrix: Matrix, tour: List[int]) -> Tuple[tuple, float]:
        """ 2-opt пробег по вершинам """
        best_change = 0
        saved = None
        size = matrix.dimension

        for n in range(size - 3):
            for m in range(n + 2, size - 1):
                i, j = tour[n], tour[m]
                x, y = tour[n + 1], tour[m + 1]
                change = matrix[i][j] + matrix[x][y]
                change -= matrix[i][x] + matrix[j][y]
                if change < best_change:
                    best_change = change
                    saved = (n, m)

        return saved, best_change

    @staticmethod
    def __swap(tour: List[int], i: int, j: int) -> List[int]:
        """ Меняем местами два элемента и разворачивает все что между ними """
        return tour[:i] + list(reversed(tour[i:j + 1])) + tour[j + 1:]
