import logging
from os import remove, path
from typing import List, Tuple
from typing import Type, Dict

import matplotlib.pyplot as plt
import numpy as np
from wget import download

from src.algorithms.lk_opt import LKOpt
from src.algorithms.lkh_opt import LKHOpt
from src.algorithms.three_opt import ThreeOpt
from src.algorithms.two_opt import TwoOpt
from src.algorithms.utils.abc_opt import AbcOpt

Edge = Tuple[int, int]
Point = Tuple[float, float]

opts_type: Dict[str, Type[AbcOpt]] = dict(two_opt=TwoOpt, three_opt=ThreeOpt, lk_opt=LKOpt, lkh_opt=LKHOpt)


def print_matrix(matrix: np.ndarray):
    """ Вывод матрицы """
    string = ''
    for s in matrix:
        for elem in s:
            string += f'{elem:0.2f}\t'
        string += '\n'
    print(string)


def to_list(points: np.ndarray) -> List[Point]:
    """ array n * 2 to List[Point], Point = float, float """
    temp = []
    for point in points:
        temp.append((point[0], point[1]))
    return temp


def to_array(points: List[Point]) -> np.ndarray:
    """ List[Point] to array, Point = float, float """
    return np.array(points, dtype=('f8', 'f8'))


def draw_tour(tour: np.ndarray, nodes: np.ndarray, color='r', show=True) -> None:
    """ Нарисовать граф по списку номеров вершин в порядке тура """
    first, second = tour[-1], tour[0]
    [x1, y1] = nodes[first]
    [x2, y2] = nodes[second]
    plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)
    for idx in range(len(tour) - 1):
        [x1, y1] = nodes[tour[idx]]
        [x2, y2] = nodes[tour[idx + 1]]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)
    if show:
        plt.show()


class TspLoader:

    @staticmethod
    def serializer(points: np.ndarray, name: str) -> None:
        with open(path.join('src', 'tsp', name), 'w') as file:
            for point in points:
                file.write(f'{point[0]} {point[1]}\n')

    @staticmethod
    def deserializer(name: str) -> np.ndarray:
        points: List[Point] = []
        with open(name, 'r') as file:
            while point := file.readline():
                temp = point.split()
                points.append((float(temp[0]), float(temp[1])))
        return to_array(points)

    @staticmethod
    def python_serializer(points: np.ndarray, name: str, directory: str) -> None:
        name = f'{name}.py'
        logging.info(name)
        points = to_list(points)
        with open(path.join(directory, name), 'w') as file:
            file.write('tsp = [\n')
            for point in points:
                file.write(f'\t{point},\n')
            file.write(']\n')

    @staticmethod
    def tsplib_deserializer(url: str, directory: str) -> np.ndarray:
        filename = download(url)
        points: List[Point] = []
        with open(filename, 'r') as file:
            for _ in range(3):
                logging.info(file.readline())
            i = int(file.readline().split()[2])
            for _ in range(2):
                file.readline()
            for _ in range(i):
                temp = file.readline().split()
                points.append((float(temp[1]), float(temp[2])))
        remove(filename)
        points: np.ndarray = to_array(points)
        TspLoader.python_serializer(points, filename.replace('.tsp', ''), directory)
        return points


def candidates(lkh: LKHOpt, nodes: np.ndarray, color='r') -> None:
    """ Граф всех кандидатов """
    temp, size = [], lkh.size
    for idx in range(0, size):
        for idy in range(idx + 1, size):
            if lkh.alpha[idx][idy] < lkh.excess:
                temp.append((idx, idy))
    for edge in temp:
        [x1, y1] = nodes[edge[0]]
        [x2, y2] = nodes[edge[1]]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)
    for idx, node in enumerate(nodes):
        plt.annotate(f'{idx}:({node[0]:0.1f},{node[1]:0.1f})', node, size=9)
    plt.show()
