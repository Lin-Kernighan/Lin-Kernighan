from random import uniform
from typing import Generator, Tuple


def generator(count, min_x=0.0, max_x=100.0, min_y=0.0, max_y=100.0) -> Generator[Tuple[float, float], None, None]:
    """ Генерируем случайную TSP — задачу. """
    for _ in range(count):
        yield uniform(min_x, max_x), uniform(min_y, max_y)
