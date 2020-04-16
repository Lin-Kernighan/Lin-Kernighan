from math import sqrt
from random import uniform
from typing import Generator, Tuple

Point = Tuple[float, float]


def generator(count: int) -> Generator[Point, None, None]:
    """ Генерируем случайную TSP — задачу. """
    max_x = max_y = sqrt(count) * 100
    for _ in range(count):
        yield uniform(0, max_x), uniform(0, max_y)
