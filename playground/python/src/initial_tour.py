from random import randrange
from sys import maxsize
from typing import List, Optional

from src.structures.graph import Graph, PoolEdges


class InitialTour:
    alpha_matrix: List[List[float]]
    weight_matrix: List[List[float]]
    selected_edges: PoolEdges

    def __init__(self, alpha_matrix: List[List[float]], weight_matrix: List[List[float]], pool: PoolEdges) -> None:
        self.alpha_matrix = alpha_matrix
        self.weight_matrix = weight_matrix
        self.selected_edges = pool

    def generate(self, best_solution: Optional[Graph]) -> Graph:
        length = len(self.alpha_matrix)
        first = previous = search = randrange(0, length - 1)  # пункт первый "choose a random node i"
        visited: List[bool] = [False] * length
        visited[first] = True

        k = 0
        new_init = Graph()
        while k < length - 1:  # вероятно не оптимально, если вообще правильно
            prices = self.alpha_matrix[previous]

            if search := self.__zero_alpha(previous, prices, visited) is not None:
                new_init.add((previous, search), self.weight_matrix[previous][search])

            elif search := self.__best_tour(previous, best_solution, visited) is not None:
                new_init.add((previous, search), self.weight_matrix[previous][search])

            elif search := self.__best_price(previous, prices, visited) is not None:
                new_init.add((previous, search), self.weight_matrix[previous][search])

            else:
                raise RuntimeError('Edge not found')

            k += 1
        new_init.add((previous, search), self.weight_matrix[first][search])
        return new_init

    def __zero_alpha(self, previous: int, prices: List[float], visited: List[bool]) -> Optional[int]:
        """ Перебираем все ребра с близостью равной нулю """
        for node, price in enumerate(prices):
            if node == previous or visited[node] or price != 0:
                continue
            if (node, previous) not in self.selected_edges:
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
