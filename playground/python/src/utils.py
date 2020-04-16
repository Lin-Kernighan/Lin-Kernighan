from typing import List
from typing import Tuple

import matplotlib.pyplot as plt

import src.structures.one_tree as one_tree
from src.structures.matrix import Matrix

Edge = Tuple[int, int]
Point = Tuple[float, float]
Node = int


def make_pair(i: int, j: int) -> Tuple[int, int]:
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
    output_list = []
    for item in range(len(tour) - num, len(tour)):
        output_list.append(tour[item])

    for item in range(0, len(tour) - num):
        output_list.append(tour[item])

    return output_list


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
