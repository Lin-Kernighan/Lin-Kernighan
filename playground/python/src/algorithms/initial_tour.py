from random import randrange, choice
from sys import maxsize
from typing import Optional, Tuple, Set, Dict

import numpy as np
from numba import njit, int64

Edge = Tuple[int, int]


class InitialTour:

    @staticmethod
    def clarke_wright(adjacency_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
        # TODO: Clarke-Wright
        pass

    @staticmethod
    def popmusic(self, adjacency_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
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
    def helsgaun(alpha_matrix: np.ndarray,
                 adjacency_matrix: np.ndarray,
                 best_solution: Optional[Set[Edge]],
                 candidates: Dict[int, list],
                 excess: float) -> Tuple[float, np.ndarray]:

        """ Генерируем новый тур """
        size, k, length = alpha_matrix.shape[0], 0, 0.0
        previous = search = randrange(0, size)  # я ищу ребро из previous в search
        visited = np.zeros(size, dtype=bool)
        order = np.zeros(size, dtype=int)
        visited[previous] = True

        while k < size - 1:
            prices = alpha_matrix[previous]
            # какая-то странная лестница получилась)
            if (search := InitialTour.__zero_alpha(previous, prices, visited)) is not None:
                length += adjacency_matrix[previous][search]
            elif (search := InitialTour.__best_tour(previous, prices, excess, best_solution, visited)) is not None:
                length += adjacency_matrix[previous][search]
            elif (search := InitialTour.__get_candidate_set(previous, candidates, visited)) is not None:
                length += adjacency_matrix[previous][search]
            elif (search := InitialTour.__just_random(previous, prices, visited)) is not None:
                length += adjacency_matrix[previous][search]
            else:
                raise RuntimeError('Edge not found')

            order[k] = previous
            visited[search] = True
            previous = search
            k += 1

        order[-1] = search
        return length, order

    @staticmethod
    @njit
    def __zero_alpha(previous: int, prices: np.ndarray, visited: np.ndarray) -> Optional[int]:
        """ Перебираем все ребра с альфа-близостью равной нулю и выбираем из них рандомное """
        zeros = np.array([idx
                          for idx, price in enumerate(prices)
                          if price == 0 and not visited[idx] and idx != previous])
        if len(zeros) != 0:
            return np.random.choice(zeros)
        return None

    @staticmethod
    def __search(edges: set, node: int) -> Set[Edge]:
        """ Ищем ребра с концом равным node """
        temp = set()
        for edge in edges:
            if node == edge[0] or node == edge[1]:
                temp.add(edge)
        return temp

    @staticmethod
    def __best_tour(previous: int,
                    prices: np.ndarray,
                    excess: float,
                    best_solution: Optional[Set[Edge]],
                    visited: np.ndarray) -> Optional[int]:
        """ Ищем в лучшем туре """

        if best_solution is None:
            return None

        search = InitialTour.__search(best_solution, previous)
        node, alpha = -1, maxsize
        for edge in search:
            temp = edge[0] if edge[0] != previous else edge[1]
            if not visited[temp] and prices[temp] < alpha < excess:
                node, alpha = temp, prices[temp]
        return node if node != -1 else None

    @staticmethod
    def __get_candidate_set(previous: int, candidates: dict, visited: np.ndarray) -> Optional[int]:
        """ Рандомный из кандидатов """
        candidates = [idx for _, _, idx in candidates[previous] if not visited[idx]]
        if candidates:
            return choice(candidates)
        return None

    @staticmethod
    @njit
    def __just_random(previous: int, prices: np.ndarray, visited: np.ndarray) -> Optional[int]:
        """ Хоть какой-нибудь """
        candidates = np.array([idx for idx, price in enumerate(prices) if not visited[idx] and previous != idx])
        if len(candidates) != 0:
            return np.random.choice(candidates)
        return None
