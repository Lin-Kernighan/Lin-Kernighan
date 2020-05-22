from os import remove, path
from typing import List, Tuple

import numpy as np
from wget import download

from src.utils import to_array, to_list

Point = Tuple[float, float]


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
        print(name)
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
                print(file.readline())
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
