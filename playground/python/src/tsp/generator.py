from math import sqrt
from random import uniform
from typing import Generator, Tuple


def generator(count) -> Generator[Tuple[float, float], None, None]:
    """ Генерируем случайную TSP — задачу. """
    min_x = min_y = 0
    max_x = max_y = sqrt(count) * 100
    for _ in range(count):
        yield uniform(min_x, max_x), uniform(min_y, max_y)
