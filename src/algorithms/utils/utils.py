from typing import Tuple, Set

import numpy as np
from numba import njit

Edge = Tuple[int, int]


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


@njit(cache=True)
def get_length(matrix: np.ndarray, tour: np.ndarray) -> float:
    """ Взятие длины по матрице смежности и туру в виде последовательных нод """
    length = matrix[tour[0]][tour[-1]]
    for idx in range(len(tour) - 1):
        length += matrix[tour[idx]][tour[idx + 1]]
    return length


def get_hash(tour: np.ndarray) -> int:
    """ Генерация хеша тура """
    return hash(str(rotate_zero(tour)))


def get_set(tour: np.ndarray) -> Set[Edge]:
    """ Генерация набора ребер тура """
    edges = set()
    for i in range(len(tour)):
        edges.add(make_pair(tour[i - 1], tour[i]))
    return edges


def right_rotate(tour: np.ndarray, num: int) -> np.ndarray:
    """ Сдвиг массива вправо на n элементов
    """
    if num == 0:
        return tour
    return np.concatenate((tour[-num:], tour[:-num]), axis=None)


def rotate_zero(tour: np.ndarray) -> np.ndarray:
    """ Проворачиваем список так, что бы первым был ноль
    """
    if tour[0] == 0:
        return tour
    return right_rotate(tour, -np.where(tour == 0)[0][0])
