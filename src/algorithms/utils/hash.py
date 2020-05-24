import numpy as np

from src.algorithms.utils.utils import rotate_zero


def generate_degrees(number: int, module: int, size: int) -> np.ndarray:
    """ Вычисление степеней 0 - size числа number по модулю module
    number: чьи степени ищем
    module: по какому модулю
    size: сколько степеней
    return: [1, number, number^2 % module ... number^(size -1)]
    """
    nums = np.zeros(size, dtype=np.int64)
    nums[0], nums[1] = 1, number
    for i in range(1, size):
        number = (number * number) % module
        nums[i] = number
    return nums


def generate_hash_from(tour: np.ndarray, degrees: np.ndarray, module: int) -> int:
    """ Вычисление хеша для тура по туру и списку степенй
    tour: список городов
    degrees: массив степеней + модуль
    return: хеш
    """
    return (tour * degrees % module).sum() % module


def generate_hash(tour: np.ndarray, number=333667, module=909090909090909091) -> int:
    """ Вычисления  хеша по туру
    tour: список вершин
    number: чьи степени будем искать
    module: по какому модулю
    return: хеш
    """
    degrees = generate_degrees(number, module, len(tour))
    return generate_hash_from(rotate_zero(tour), degrees, module)
