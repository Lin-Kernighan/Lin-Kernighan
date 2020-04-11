from random import uniform
from typing import Tuple


def generator(count, min_x=0.0, max_x=100.0, min_y=0.0, max_y=100.0) -> Tuple[float, float]:
    temp = 0
    while temp < count:
        temp += 1
        yield uniform(min_x, max_x), uniform(min_y, max_y)
