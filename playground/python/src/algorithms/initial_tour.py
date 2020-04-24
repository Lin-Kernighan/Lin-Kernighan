from random import randrange, choice
from sys import maxsize
from typing import List, Optional, Tuple

import numpy as np
from numba import njit, int64

from src.structures.graph import Graph


class InitialTour:

    @staticmethod
    def clarke_wright(adjacency_matrix: np.ndarray) -> List[int]:
        # TODO: Clarke-Wright
        pass

    @staticmethod
    def popmusic(self, adjacency_matrix: np.ndarray) -> List[int]:
        # TODO: POPMUSIC
        pass

    @staticmethod
    @njit
    def greedy(matrix: np.ndarray, point: Optional[int] = None) -> Tuple[float, np.ndarray]:
        """ Строим жадным методом """
        length = matrix.shape[0]
        start = previous = point if point is not None else randrange(0, length)  # я ищу ребро из previous в search
        search, path, k = 0, 0.0, 0

        visited = np.zeros((length,), dtype=int64)
        order = np.zeros((length,), dtype=int64)
        visited[previous] = 1

        while k < length - 1:
            minimum, search = maxsize, -1
            for idx, price in enumerate(matrix[previous]):
                if idx != previous and minimum > price and visited[idx] == 0:
                    minimum, search = price, idx
            visited[search] = 1
            path += minimum
            order[k] = previous
            previous = search
            k += 1

        path += matrix[search][start]
        order[-1] = search
        return path, order

    @staticmethod
    def helsgaun(alpha_matrix: np.ndarray, best_solution: Optional[Graph], excess: float) -> List[int]:
        """ Генерируем новый тур """
        length = len(alpha_matrix)
        previous = search = randrange(0, length)  # я ищу ребро из previous в search
        visited: List[bool] = [False] * length
        order: List[int] = [0] * length
        visited[previous] = True

        k = 0
        while k < length - 1:  # вероятно не оптимально, если вообще правильно
            prices = alpha_matrix[previous]
            # какая-то странная лестница получилась)
            if (search := InitialTour.__zero_alpha(previous, prices, visited)) is not None:
                pass
            elif (search := InitialTour.__best_tour(previous, prices, excess, best_solution, visited)) is not None:
                pass
            elif (search := InitialTour.__get_candidate_set(previous, prices, visited, excess)) is not None:
                pass
            elif (search := InitialTour.__just_random(previous, prices, visited)) is not None:
                pass
            else:
                raise RuntimeError('Edge not found')

            order[k] = previous
            visited[search] = True
            previous = search
            k += 1
        order[-1] = search
        return order

    @staticmethod
    def __zero_alpha(previous: int, prices: List[float], visited: List[bool]) -> Optional[int]:
        """ Перебираем все ребра с альфа-близостью равной нулю и выбираем из них рандомное """
        zeros = [idx
                 for idx, price in enumerate(prices)
                 if price == 0 and not visited[idx] and idx != previous]
        if zeros:
            return choice(zeros)
        return None

    @staticmethod
    def __best_tour(previous: int,
                    prices: List[float],
                    excess: float,
                    best_solution: Optional[Graph],
                    visited: List[bool]) -> Optional[int]:
        """ Ищем в лучшем туре """

        if best_solution is None:
            return None

        search = best_solution.search(previous)
        node, alpha = -1, maxsize
        for edge in search:
            temp = edge[0] if edge[0] != previous else edge[1]
            if not visited[temp] and prices[temp] < alpha < excess:
                node, alpha = temp, prices[temp]
        return node if node != -1 else None

    @staticmethod
    def __get_candidate_set(previous: int, prices: List[float], visited: List[bool], excess: float) -> Optional[int]:
        """ Рандомный из кандидатов """
        candidates = [idx
                      for idx, price in enumerate(prices)
                      if price < excess and idx != previous and not visited[idx]]
        if candidates:
            return choice(candidates)
        return None

    @staticmethod
    def __just_random(previous: int, prices: List[float], visited: List[bool]) -> Optional[int]:
        """ Хоть какой-нибудь """
        candidates = [idx for idx, price in enumerate(prices) if not visited[idx] and previous != idx]
        if candidates:
            return choice(candidates)
        return None
