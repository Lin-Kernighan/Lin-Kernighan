import numba as nb
import numpy as np

from src.algorithms.utils.utils import rotate_zero


@nb.njit(cache=True)
def generate_degrees(number: int, module: int, size: int) -> np.ndarray:
    """ Вычисление степеней 0 - size числа number по модулю module
    number: чьи степени ищем
    module: по какому модулю
    size: сколько степеней
    return: [1, number, number^2 % module ... number^(size -1), module]
    """
    nums = np.zeros(size + 1, dtype=nb.int64)
    nums[-1], nums[0], nums[1] = module, 1, number
    for i in range(1, size):
        number = (number * number) % module
        nums[i] = number
    return nums


@nb.njit(cache=True)
def generate_hash_from(tour: np.ndarray, degrees: np.ndarray) -> int:
    """ Вычисление хеша для тура по туру и списку степенй
    tour: список городов
    degrees: массив степеней + модуль
    return: хеш
    """
    h, module = 0, degrees[-1]
    for i, num in enumerate(tour):
        h += (degrees[i] * num) % module
        h = h % module
    return h


@nb.njit(cache=True)
def generate_hash(tour: np.ndarray, number=38917, module=1000000007) -> int:
    """ Вычисления  хеша по туру
    tour: список вершин
    number: чьи степени будем искать
    module: по какому модулю
    return: хеш
    """
    degrees = generate_degrees(number, module, len(tour))
    return generate_hash_from(rotate_zero(tour), degrees)
