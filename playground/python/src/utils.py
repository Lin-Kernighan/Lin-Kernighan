from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from pandas import DataFrame

Edge = Tuple[int, int]
Point = Tuple[float, float]
Node = int


@njit(cache=True)
def between(tour: np.ndarray, first: int, second: int, third: int) -> bool:
    if first < second:  # [ ... start -> ... search ... <- end ... ]
        if first < third < second:
            return True
    else:  # [ ? ... <- end ... start -> ... ? ]
        if 0 <= third < second or first < third < len(tour):
            return True
    return False


@njit(cache=True)
def around(tour: np.ndarray, it: int) -> tuple:
    s = len(tour)
    return ((it + 1) % s, tour[(it + 1) % s]), ((it - 1) % s, tour[(it - 1) % s])


@njit(cache=True)
def swap(tour: np.ndarray, x: int, y: int) -> np.ndarray:
    """ Переворот куска тура: [x, y], включительно!
    tour: список городов
    x, y: индексы
    return: измененный список
    """
    size, temp = len(tour), 0
    if x < y:
        temp = (y - x + 1) // 2
    elif x > y:
        temp = ((size - x) + y + 2) // 2
    for i in range(temp):
        first, second = (x + i) % size, (y - i) % size
        tour[first], tour[second] = tour[second], tour[first]
    return tour


@njit(cache=True)
def make_pair(i: int, j: int) -> Edge:
    """ Правильная пара для упрощения хранения ребер """
    return (i, j) if i > j else (j, i)


@njit(cache=True)
def check_dlb(dlb: np.ndarray, idx: int) -> bool:
    """ Проверка don't look bites
    idx: индекс города в туре
    """
    s = len(dlb)
    if s != 1 and dlb[idx] and dlb[(idx - 1) % s] and dlb[(idx + 1) % s]:
        return True
    return False


def get_hash(tour: np.ndarray) -> int:
    """ хеш тура """
    return hash(str(rotate_zero(tour)))


def print_matrix(matrix: np.ndarray):
    """ Вывод матрицы """
    string = ''
    for s in matrix:
        for elem in s:
            string += f'{elem:0.2f}\t'
        string += '\n'
    print(string)


def get_set(tour: np.ndarray) -> set:
    """ ребра тура """
    edges = set()
    for i in range(len(tour)):
        edges.add(make_pair(tour[i - 1], tour[i]))
    return edges


def to_list(points: np.ndarray) -> List[Point]:
    """ array n * 2 to List[Point], Point = float, float """
    temp = []
    for point in points:
        temp.append((point[0], point[1]))
    return temp


def to_array(points: List[Point]) -> np.ndarray:
    """ List[Point] to array, Point = float, float """
    return np.array(points, dtype=('f8', 'f8'))


@njit(cache=True)
def get_length(matrix: np.ndarray, tour: np.ndarray) -> float:
    """ Взятие длины по матрице смежности и туру в виде последовательных нод """
    length = matrix[tour[0]][tour[-1]]
    for idx in range(len(tour) - 1):
        length += matrix[tour[idx]][tour[idx + 1]]
    return length


def right_rotate(tour: np.ndarray, num: int) -> np.ndarray:
    """ Сдвиг массива вправо на n элементов
    Костыль на крайние случаи масиива, чтобы алгоритм их тоже проверил
    Можно сделать иначе: но нужно переписать swap и взять индекс по модулю в алгоритме
    """
    if num == 0:
        return tour
    return np.concatenate((tour[-num:], tour[:-num]), axis=None)


def rotate_zero(tour: np.ndarray) -> np.ndarray:
    """ Проворачиваем список так, что бы первым был ноль """
    if tour[0] == 0:
        return tour
    return right_rotate(tour, -np.where(tour == 0)[0][0])


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


def draw_tour(tour: np.ndarray, nodes: np.ndarray, color: str) -> None:
    """ Нарисовать граф по списку номеров вершин в порядке тура """
    first, second = tour[-1], tour[0]
    [x1, y1] = nodes[first]
    [x2, y2] = nodes[second]
    plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)
    for idx in range(len(tour) - 1):
        [x1, y1] = nodes[tour[idx]]
        [x2, y2] = nodes[tour[idx + 1]]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)


def candidates(alpha_matrix: np.ndarray, nodes: np.ndarray, excess: float, color: str) -> None:
    """ Граф всех кандидатов """
    temp = []
    for idx in range(0, alpha_matrix.shape[0]):
        for idy in range(idx + 1, alpha_matrix.shape[0]):
            if alpha_matrix[idx][idy] < excess:
                temp.append((idx, idy))

    for edge in temp:
        [x1, y1] = nodes[edge[0]]
        [x2, y2] = nodes[edge[1]]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)

    for idx, node in enumerate(nodes):
        plt.annotate(f'{idx}:({node[0]:0.1f},{node[1]:0.1f})', node, size=9)
