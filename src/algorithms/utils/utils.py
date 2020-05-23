from typing import Tuple, Set

import numba as nb
import numpy as np

Edge = Tuple[int, int]


@nb.njit(cache=True)
def between(size: int, first: int, second: int, third: int) -> bool:
    """ Проверка находится ли third между first и second
    size: размер тура
    first, second, third: индексы вершин
    return: между?
    """
    if first < second:  # [ ... start -> ... search ... <- end ... ]
        if first < third < second:
            return True
    else:  # [ ? ... <- end ... start -> ... ? ]
        if 0 <= third < second or first < third < size:
            return True
    return False


@nb.njit(cache=True)
def around(tour: np.ndarray, it: int) -> tuple:
    """ Возвращает predecessor and successor вершины t
    tour: список городов
    it: индекс вырешины t
    return: ((index suc, suc), (index pred, pred))
    """
    s = len(tour)
    return ((it + 1) % s, tour[(it + 1) % s]), ((it - 1) % s, tour[(it - 1) % s])


@nb.njit(cache=True)
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


@nb.njit(cache=True)
def make_pair(i: int, j: int) -> Edge:
    """ Правильная пара для упрощения хранения ребер
    i, j: индексы ребер
    return: правильное ребро
    """
    return (i, j) if i > j else (j, i)


@nb.njit(cache=True)
def check_dlb(dlb: np.ndarray, idx: int) -> bool:
    """ Проверка don't look bites
    dlb: массив dlb
    idx: индекс города в туре
    return: пропускать или нет
    """
    s = len(dlb)
    if s != 1 and dlb[idx] and dlb[(idx - 1) % s] and dlb[(idx + 1) % s]:
        return True
    return False


@nb.njit(cache=True)
def get_length(tour: np.ndarray, matrix: np.ndarray) -> float:
    """ Взятие длины по матрице смежности и туру в виде последовательных вершин
    tour: список вершин
    matrix: матрица весов
    return: длина
    """
    length = matrix[tour[0]][tour[-1]]
    for idx in range(len(tour) - 1):
        length += matrix[tour[idx]][tour[idx + 1]]
    return length


def get_set(tour: np.ndarray) -> Set[Edge]:
    """ Генерация набора ребер тура
    tour: список вершин
    return: set из ребер
    """
    edges = set()
    for i in range(len(tour)):
        edges.add(make_pair(tour[i - 1], tour[i]))
    return edges


@nb.njit(cache=True)
def rotate(tour: np.ndarray, num: int) -> np.ndarray:
    """ Сдвиг массива влево на n элементов
    tour: список вершин
    num: на сколько двигаем
    return: сдвинутый
    """
    if num == 0:
        return tour
    size, idx = len(tour), 0
    temp = np.zeros(size, dtype=nb.int64)
    for i in range(num, size):
        temp[idx] = tour[i]
        idx += 1
    for j in range(0, num):
        temp[idx] = tour[j]
        idx += 1
    return temp


@nb.njit(cache=True)
def rotate_zero(tour: np.ndarray) -> np.ndarray:
    """ Проворачиваем список так, что бы первым был ноль
    tour: список вершин
    return: свдинутый список
    """
    if tour[0] == 0:
        return tour
    return rotate(tour, np.where(tour == 0)[0][0])
