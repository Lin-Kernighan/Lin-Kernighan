from typing import List, Tuple

from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import Matrix


def right_rotate(tour: list, num: int) -> list:
    """ Сдвиг массива вправо на n
    Костыль на крайние случаи масиива, чтобы алгоритм их тоже проверил
    Можно сделать иначе: но нужно переписать swap и взятие по модулю в алгоритме
    """
    output_list = []
    for item in range(len(tour) - num, len(tour)):
        output_list.append(tour[item])

    for item in range(0, len(tour) - num):
        output_list.append(tour[item])

    return output_list


class TwoOpt:

    @staticmethod
    def run(points: List[Tuple[float, float]]) -> List[int]:
        """ Полный запуск на точках """
        matrix = Matrix.weight_matrix(points)
        init_tour = InitialTour.greedy(matrix)
        return TwoOpt.optimize(init_tour, matrix)

    @staticmethod
    def optimize(tour: List[int], matrix: Matrix) -> List[int]:
        """ Запуск на готовом туре и матрице смежностей """
        best_change = -1
        exchange = 0
        while best_change < 0:
            best_change, tour = TwoOpt.__two_opt(matrix, tour)
            if best_change == 0:
                tour = right_rotate(tour, len(tour) // 3)  # костылек, зато пока работает быстрее чем алгоритм, забью
                best_change, tour = TwoOpt.__two_opt(matrix, tour)
            print(f'{exchange}\t:\t{-best_change}')  # оставлю, чтобы видеть, что алгоритм не помер еще
            exchange += 1
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
    def __swap(tour: list, i: int, j: int) -> list:
        """ Меняем местами два элемента и разворачивает все что между ними """
        return tour[:i] + list(reversed(tour[i:j + 1])) + tour[j + 1:]
