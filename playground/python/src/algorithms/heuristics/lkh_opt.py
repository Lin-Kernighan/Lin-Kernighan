from collections import defaultdict
from typing import Tuple, Set, Optional

import numpy as np

from src.algorithms.initial_tour import InitialTour
from src.algorithms.subgradient_optimization import SubgradientOptimization
from src.structures.collector import Collector
from src.structures.matrix import alpha_matrix
from src.structures.one_tree import one_tree_topology
from src.structures.tour.list_tour import ListTour
from src.utils import make_pair

Edge = Tuple[int, int]
Node = int


class LkhOpt:

    def __init__(self, adjacency: np.ndarray, excess: float = None):
        self.size, self.matrix = adjacency.shape[0], adjacency
        self.gradient = SubgradientOptimization.run(adjacency)
        SubgradientOptimization.make_move(self.gradient.pi_sum, self.matrix)
        length, f, s, topology = one_tree_topology(self.matrix)
        self.alpha = alpha_matrix(self.matrix, f, s, topology)
        self.excess = excess if excess is not None else 1 / self.size * length

        self.candidates = defaultdict(list)
        for i in range(0, self.size):
            for j in range(i + 1, self.size):
                if self.alpha[i][j] < self.excess:
                    self.candidates[i].append((self.alpha[i][j], j))
                    self.candidates[j].append((self.alpha[i][j], i))
        for _, candidate in self.candidates.items():
            candidate.sort()

        SubgradientOptimization.get_back(self.gradient.pi_sum, self.matrix)
        self.length, self.tour = InitialTour.helsgaun(self.alpha, self.matrix, None, self.candidates, excess)
        self.temp_length = self.length

        self.solutions: Set[int] = set()
        self.best_solution: Optional[np.ndarray] = None
        self.collector: Optional[Collector] = None

    def optimize(self):
        iteration, self.collector = 0, Collector(['length', 'gain'], {'lkh': self.size})
        self.collector.update({'length': self.length, 'gain': 0})

        better = True
        while better:
            print(f'{iteration} : {self.length}')
            better = self.improve()
            gain = self.length - self.temp_length
            self.length = self.temp_length
            self.collector.update({'length': self.length, 'gain': gain})
            self.solutions.add(hash(str(self.tour)))
            iteration += 1

        return self.tour

    def improve(self) -> bool:
        tour = ListTour(self.tour)

        for index in range(len(tour)):
            t1 = tour[index]
            around = tour.around(t1)

            for t2 in around:
                broken = {make_pair(t1, t2)}
                gain = self.matrix[t1][t2]
                close = self.closest(t2, tour, gain, broken, set())
                tries = 5

                for t3, (_, curr_gain) in close:
                    if t3 in around:
                        continue

                    joined = {make_pair(t2, t3)}
                    if self.choose_x(tour, t1, t3, curr_gain, broken, joined):
                        return True

                    tries -= 1
                    if tries == 0:
                        break

        return False

    def closest(self, t2i: Node, tour: ListTour, gain: float, broken: Set[Edge], joined: Set[Edge]) -> list:
        neighbours = {}

        # Create the neighbours of t_2i
        for cost, node in self.candidates[t2i]:  # и вот для того t2 берем всех соседей
            yi = make_pair(t2i, node)  # фигачим ребро
            curr_gain = gain - self.matrix[t2i][node]  # выигрывает что-то на этом

            # Any new edge has to have a positive running sum, not be a broken
            # edge and not belong to the tour.
            if curr_gain <= 0 or yi in broken or yi in tour:  # очевидно
                continue

            for successor in tour.around(node):  # и вот для соседа t2 мы берем соседа
                xi = make_pair(node, successor)  # фигачим под него ребро
                if xi not in broken and xi not in joined:  # тааак
                    # и вот diff для сосед t2 и сосед соседа t2 и t2 и сосед соседа t2
                    diff = self.matrix[node][successor] - self.matrix[t2i][node]
                    if node in neighbours and diff > neighbours[node][0]:
                        # видимо, если есть круче diff, то запоминаем
                        neighbours[node][0] = diff
                    else:
                        # если в очередном словаре не нашлось, сохраняем
                        # по ключу сосед соседа diff и curr_gain
                        neighbours[node] = [diff, curr_gain]

        # Sort the neighbours by potential gain
        return sorted(neighbours.items(), key=lambda x: x[1][0], reverse=True)

    def choose_x(self, tour: ListTour, t1: Node, last: Node, gain: float, broken: Set[Edge], joined: Set[Edge]) -> bool:
        """
        Choose an edge to omit from the tour.
        Parameters:
            - tour: current tour to optimise
            - t1: starting node for the current k-opt
            - last: tail of the last edge added (t_2i-1)
            - gain: current gain (curr_gain)
            - broken: potential edges to remove (X)
            - joined: potential edges to add (Y)
        Return: whether we found an improved tour
        """
        if len(broken) == 4:
            pred, suc = tour.around(last)

            # Give priority to the longest edge for x_4  # почему? зачем?
            if self.matrix[pred][last] > self.matrix[suc][last]:
                around = [pred]
            else:
                around = [suc]
        else:
            around = tour.around(last)

        for t2i in around:
            xi = make_pair(last, t2i)
            # Gain at current iteration
            curr_gain = gain + self.matrix[last][t2i]

            # Verify that X and Y are disjoint, though I also need to check
            # that we are not including an x_i again for some reason.
            if xi not in joined and xi not in broken:
                added = joined.copy()
                removed = broken.copy()

                removed.add(xi)
                added.add(make_pair(t2i, t1))  # Try to relink the tour

                relink = curr_gain - self.matrix[t2i][t1]
                is_tour, new_tour = tour.generate(removed, added)

                # The current solution does not form a valid tour
                if not is_tour and len(added) > 2:
                    continue

                # Stop the search if we come back to the same solution
                if str(new_tour) in self.solutions:
                    return False

                # Save the current solution if the tour is better, we need
                # `is_tour` again in the case where we have a non-sequential
                # exchange with i = 2
                if is_tour and relink > 0:
                    self.tour = new_tour
                    self.temp_length -= relink
                    return True
                else:
                    # Pass on the newly "removed" edge but not the relink
                    choice = self.choose_y(tour, t1, t2i, curr_gain, removed, joined)

                    if len(broken) == 2 and choice:
                        return True
                    else:
                        # Single iteration for i > 2
                        return choice

        return False

    def choose_y(self, tour: ListTour, t1: Node, t2i: Node, gain: float, broken: Set[Edge], joined: Set[Edge]):
        """ Choose an edge to add to the new tour.
        Parameters:
            - tour: current tour to optimise
            - t1: starting node for the current k-opt
            - t2i: tail of the last edge removed (t_2i)
            - gain: current gain (Gi)
            - broken: potential edges to remove (X)
            - joined: potential edges to add (Y)
        Return: whether we found an improved tour
        """
        ordered = self.closest(t2i, tour, gain, broken, joined)

        if len(broken) == 2:
            # Check the five nearest neighbours when i = 2
            top = 5
        else:
            # Otherwise the closest only
            top = 1

        for node, (_, curr_gain) in ordered:
            yi = make_pair(t2i, node)
            added = joined.copy()
            added.add(yi)

            # Stop at the first improving tour
            if self.choose_x(tour, t1, node, curr_gain, broken, added):
                return True

            top -= 1
            # Tried enough options
            if top == 0:
                return False

        return False
