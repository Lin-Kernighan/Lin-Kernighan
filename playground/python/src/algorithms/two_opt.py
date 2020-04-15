from typing import List

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


def swap(tour: list, i: int, j: int) -> list:
    """ Меняем местами два элемента и разворачивает все что между ними """
    return tour[:i] + list(reversed(tour[i:j + 1])) + tour[j + 1:]


class TwoOpt:

    @staticmethod
    def run(matrix: Matrix, init_tour: List[int]) -> List[int]:
        return TwoOpt.optimize(init_tour, matrix)

    @staticmethod
    def optimize(tour: List[int], matrix: Matrix) -> List[int]:
        best_change = -1

        while best_change < 0:
            saved, best_change = TwoOpt.improve(matrix, tour)
            if best_change < 0:
                i, j = saved
                tour = swap(tour, i + 1, j)
            tour = right_rotate(tour, 2)
        return tour

    @staticmethod
    def improve(matrix: Matrix, tour: List[int]):
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
