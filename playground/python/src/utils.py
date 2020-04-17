from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
from numba import njit

import src.structures.one_tree as one_tree
from src.structures.matrix import Matrix

Edge = Tuple[int, int]
Point = Tuple[float, float]
Node = int


@njit
def make_pair(i: int, j: int) -> Edge:
    """ Правильная пара для упрощения хранения ребер """
    return (i, j) if i > j else (j, i)


def get_length(matrix: Matrix, tour: List[int]) -> float:
    """ Взятие длины по матрице смежности и туру в виде последовательных нод """
    length = matrix[tour[0]][tour[-1]]
    for idx in range(len(tour) - 1):
        length += matrix[tour[idx]][tour[idx + 1]]
    return length


def right_rotate(tour: list, num: int) -> list:
    """ Сдвиг массива вправо на n элементов
    Костыль на крайние случаи масиива, чтобы алгоритм их тоже проверил
    Можно сделать иначе: но нужно переписать swap и взять индекс по модулю в алгоритме
    """
    if num == 0:
        return tour
    return tour[-num:] + tour[:-num]


def rotate_zero(tour: List[int]) -> list:
    """ Проворачиваем список так, что бы первым был ноль """
    if tour[0] == 0:
        return tour
    return right_rotate(tour, -tour.index(0))


def draw_edges(edges: List[one_tree.Edge], nodes: List[Point], color: str) -> None:
    """ Нарисовать граф по списку ребер типа Edge """
    for edge in edges:
        [x1, y1] = nodes[edge.dst]
        [x2, y2] = nodes[edge.src]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)


def draw_tour(tour: List[int], nodes: List[Point], color: str) -> None:
    """ Нарисовать граф по списку номеров вершин в порядке тура """
    first, second = tour[-1], tour[0]
    [x1, y1] = nodes[first]
    [x2, y2] = nodes[second]
    plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)
    for idx in range(len(tour) - 1):
        [x1, y1] = nodes[tour[idx]]
        [x2, y2] = nodes[tour[idx + 1]]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)


def candidates(alpha_matrix: Matrix, nodes: List[Point], excess: float, color: str) -> None:
    """ Граф всех кандидатов """
    temp = []
    for idx in range(0, alpha_matrix.dimension):
        for idy in range(idx + 1, alpha_matrix.dimension):
            if alpha_matrix[idx][idy] < excess:
                temp.append((idx, idy))
    print(len(temp))

    for edge in temp:
        [x1, y1] = nodes[edge[0]]
        [x2, y2] = nodes[edge[1]]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)

    for idx, node in enumerate(nodes):
        plt.annotate(f'{idx}:({node[0]:0.1f},{node[1]:0.1f})', node, size=9)
