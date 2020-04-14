from random import randrange, choice
from sys import maxsize
from typing import List, Optional

from src.structures.graph import Graph
from src.structures.matrix import Matrix


class InitialTour:

    @staticmethod
    def clarke_wright(weight_matrix: Matrix) -> List[int]:
        # TODO: clarke wright
        # point = randrange(0, len(weight_matrix))
        # return InitialTour.greedy(Matrix.savings_matrix(weight_matrix, point), point)
        pass

    @staticmethod
    def greedy(matrix: Matrix, point: Optional[int] = None) -> List[int]:
        length = len(matrix)
        previous = point if point is not None else randrange(0, length)  # я ищу ребро из previous в search
        search = 0

        visited: List[bool] = [False] * length
        order: List[int] = [0] * length
        visited[previous] = True

        k = 0
        while k < length - 1:
            minimum, search = maxsize, -1
            for idx, price in enumerate(matrix[previous]):
                if idx != previous and minimum > price and not visited[idx]:
                    minimum, search = price, idx
            visited[search] = True
            order[k] = previous
            previous = search
            k += 1
        order[-1] = search
        return order

    @staticmethod
    def helsgaun(alpha_matrix: Matrix, best_solution: Optional[Graph], excess: Optional[float]) -> List[int]:
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
