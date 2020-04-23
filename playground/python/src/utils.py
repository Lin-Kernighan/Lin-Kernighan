from typing import List, Tuple

import matplotlib.pyplot as plt
from numba import njit
from numpy import ndarray
from pandas import DataFrame

import src.structures.one_tree as one_tree

Edge = Tuple[int, int]
Point = Tuple[float, float]
Node = int


@njit
def make_pair(i: int, j: int) -> Edge:
    """ Правильная пара для упрощения хранения ребер """
    return (i, j) if i > j else (j, i)


def print_matrix(matrix: ndarray):
    """ Вывод матрицы """
    string = ''
    for s in matrix:
        for elem in s:
            string += f'{elem:0.2f}\t'
        string += '\n'
    print(string)


def get_length(matrix: ndarray, tour: List[int]) -> float:
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


def draw_plots_i_y(data: List[DataFrame], names: List[str], columns: List[str], file: str, directory: str) -> None:
    """ Рисуем и сохраняем много графиков column == y_name от итерации """
    for column in columns:
        frame = DataFrame()
        for i in range(len(data)):
            frame[names[i]] = data[i][column]
        fig = frame.plot().get_figure()
        plt.ylabel(column)
        plt.xlabel('iteration')
        plt.show()
        if file is not None:
            fig.savefig(f'{directory}/{file}_{column}.png')


def draw_plot_x_y(data: List[DataFrame], names: List[str], x_name: str, y_name: str, file: str, directory: str) -> None:
    """ Рисуем и сохраняем график x_name от y_name """
    for i in range(len(data)):
        plt.plot(data[i][x_name], data[i][y_name], label=names[i])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    if file is not None:
        plt.savefig(f'{directory}/{file}_{x_name}_{y_name}.png')
    plt.show()


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


def candidates(alpha_matrix: ndarray, nodes: List[Point], excess: float, color: str) -> None:
    """ Граф всех кандидатов """
    temp = []
    for idx in range(0, alpha_matrix.shape[0]):
        for idy in range(idx + 1, alpha_matrix.shape[0]):
            if alpha_matrix[idx][idy] < excess:
                temp.append((idx, idy))
    print(len(temp))

    for edge in temp:
        [x1, y1] = nodes[edge[0]]
        [x2, y2] = nodes[edge[1]]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)

    for idx, node in enumerate(nodes):
        plt.annotate(f'{idx}:({node[0]:0.1f},{node[1]:0.1f})', node, size=9)
