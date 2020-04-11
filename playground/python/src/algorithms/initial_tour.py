from dataclasses import dataclass
from random import randrange
from sys import maxsize
from typing import List, Optional

from src.structures.graph import Graph
from src.structures.matrix import Matrix


@dataclass
class InitialTour:
    alpha_matrix: Matrix
    weight_matrix: Matrix

    def generate(self, best_solution: Optional[Graph]) -> List[int]:
        """ Генерируем новый тур """
        length = len(self.alpha_matrix)
        first = previous = search = randrange(0, length - 1)  # пункт первый "choose a random node i"
        visited: List[bool] = [False] * length
        order: List[int] = [0] * length
        visited[first] = True

        k = 0
        while k < length - 1:  # вероятно не оптимально, если вообще правильно
            prices = self.alpha_matrix[previous]

            if (search := self.__zero_alpha(previous, prices, visited)) is not None:
                pass
            elif (search := self.__best_tour(previous, best_solution, visited)) is not None:
                pass
            elif (search := self.__best_price(previous, prices, visited)) is not None:
                pass
            else:
                raise RuntimeError('Edge not found')

            order[k] = previous
            visited[search] = True
            previous = search
            k += 1
        order[-1] = search
        return order

    def __zero_alpha(self, previous: int, prices: List[float], visited: List[bool]) -> Optional[int]:
        """ Перебираем все ребра с альфа-близостью равной нулю """
        for node, price in enumerate(prices):
            if node == previous or visited[node] or price != 0:
                continue
            return node
        return None

    def __best_tour(self, previous: int, best_solution: Optional[Graph], visited: List[bool]) -> Optional[int]:
        """ Ищем в лучшем туре """
        if best_solution is None:
            return None

        search = best_solution.search(previous)
        node, alpha = -1, maxsize
        for edge in search:
            temp = edge[0] if edge[0] != previous else edge[1]
            if not visited[temp] and alpha > self.alpha_matrix[previous][temp]:
                node, alpha = temp, self.alpha_matrix[previous][temp]
        return node if node != -1 else None

    def __best_price(self, previous: int, prices: List[float], visited: List[bool]) -> Optional[int]:
        """ Просто лучший по альфа-близости """
        node, alpha = -1, maxsize
        for index, price in enumerate(prices):
            if index == previous or visited[index]:
                continue

            if not visited[index] and alpha > self.alpha_matrix[previous][index]:
                node, alpha = index, alpha > self.alpha_matrix[previous][index]
        return node if node != -1 else None
