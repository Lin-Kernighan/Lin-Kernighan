from collections import defaultdict
from typing import Tuple, Set, Optional

import numpy as np

from src.algorithms.initial_tour import InitialTour
from src.algorithms.subgradient_optimization import SubgradientOptimization
from src.structures.collector import Collector
from src.structures.matrix import alpha_matrix
from src.structures.one_tree import one_tree_topology
from src.structures.tabu_list import TabuSet
from src.structures.tour.list_tour import ListTour
from src.utils import make_pair, get_hash, get_set

Edge = Tuple[int, int]
Node = int


class LkhOpt:

    def __init__(self, adjacency: np.ndarray, dlb=True, excess: float = None):
        self.size, self.matrix = adjacency.shape[0], adjacency
        self.gradient = SubgradientOptimization.run(adjacency)
        SubgradientOptimization.make_move(self.gradient.pi_sum, self.matrix)
        length, f, s, self.best_solution, topology = one_tree_topology(self.matrix)
        self.excess = excess if excess is not None else 1 / self.size * length
        self.alpha = alpha_matrix(self.matrix, f, s, topology)
        SubgradientOptimization.get_back(self.gradient.pi_sum, self.matrix)

        self.candidates = defaultdict(list)
        for i in range(0, self.size):
            for j in range(i + 1, self.size):
                if self.alpha[i][j] < self.excess:
                    self.candidates[i].append((self.alpha[i][j], self.matrix[i][j], j))
                    self.candidates[j].append((self.alpha[i][j], self.matrix[i][j], i))
        for _, candidate in self.candidates.items():
            candidate.sort()

        self.length, self.tour = InitialTour.greedy(self.matrix)
        self.temp_length = self.length

        self.solutions: Set[int] = set()
        self.collector: Optional[Collector] = None
        self.dlb = np.zeros(self.size, dtype=bool) if dlb else None

    def optimize(self) -> np.ndarray:
        iteration, self.collector = 0, Collector(['length', 'gain'], {'lkh': self.size})
        self.collector.update({'length': self.length, 'gain': 0})

        better = True
        while better:
            print(f'{iteration} : {self.length}')
            better = self.improve()
            gain = self.length - self.temp_length
            self.length = self.temp_length
            self.collector.update({'length': self.length, 'gain': gain})
            self.solutions.add(get_hash(self.tour))
            iteration += 1

        return self.tour

    def tabu_optimize(self, tabu_list: TabuSet, collector: Collector) -> np.ndarray:
        self.solutions = self.solutions | tabu_list.data
        self.collector = collector
        self.collector.update({'length': self.length, 'gain': 0})

        better = True
        while better:
            better = self.improve()
            gain = self.length - self.temp_length
            self.length = self.temp_length
            self.solutions.add(get_hash(self.tour))
            self.collector.update({'length': self.length, 'gain': gain})

        return self.tour

    def lkh_optimize(self, iterations=10) -> np.ndarray:
        self.optimize()
        best_length, best_tour = self.length, self.tour
        best_solution = get_set(self.tour)

        for _ in range(iterations):
            self.dlb = np.zeros(self.size, dtype=bool) if self.dlb is not None else None
            self.length, self.tour = \
                InitialTour.helsgaun(self.alpha, self.matrix, best_solution, self.candidates, self.excess)
            self.temp_length = self.length
            self.optimize()
            if self.length < best_length:
                best_length, best_tour = self.length, self.tour
                best_solution = get_set(self.tour)

        self.length, self.tour = best_length, best_tour
        return self.tour

    def improve(self) -> bool:
        tour = ListTour(self.tour)

        for index in range(self.size):
            t1 = tour[index]
            if self.dlb is not None \
                    and self.dlb[t1] and self.dlb[(t1 - 1) % self.size] and self.dlb[(t1 + 1) % self.size]:
                continue

            around = tour.around(t1)

            for t2 in around:
                broken = {make_pair(t1, t2)}

                # LKH rule(1)
                if broken in self.best_solution:
                    continue

                gain = self.matrix[t1][t2]
                close = self.closest(t2, tour, gain, broken, set())

                for t3, (_, curr_gain) in close:
                    if t3 in around:
                        continue

                    joined = {make_pair(t2, t3)}
                    if self.choose_x(tour, t1, t3, curr_gain, broken, joined):

                        if self.dlb is not None:
                            for x, y in broken:
                                self.dlb[x] = self.dlb[y] = False
                            for x, y in joined:
                                self.dlb[x] = self.dlb[y] = False

                        return True

            if self.dlb is not None:
                self.dlb[t1] = True

        return False

    def closest(self, t2i: Node, tour: ListTour, gain: float, broken: Set[Edge], joined: Set[Edge]) -> list:
        """ Find the closest neighbours of a node ordered by potential gain.
        As a side-effect, also compute the partial improvement of joining a node.
        t2i: node to relink from
        tour: current tour to optimise
        gain: current gain
        broken: set of edges to remove (X)
        joined: set of edges to join (Y)
        return: sorted list of neighbours based on potential improvement with next omission
        """
        neighbours = {}

        for _, _, node in self.candidates[t2i]:
            yi = make_pair(t2i, node)
            curr_gain = gain - self.matrix[t2i][node]

            # LK rule(2), LK rule(4)
            if curr_gain <= 0 or yi in broken or yi in tour:
                continue

            for successor in tour.around(node):
                xi = make_pair(node, successor)

                # LK rule(4)
                if xi not in broken and xi not in joined:
                    diff = self.matrix[node][successor] - self.matrix[t2i][node]

                    if node in neighbours and diff > neighbours[node][0]:
                        neighbours[node][0] = diff
                    else:
                        neighbours[node] = [diff, curr_gain]

        return sorted(neighbours.items(), key=lambda x: x[1][0], reverse=True)

    def choose_x(self, tour: ListTour, t1: Node, last: Node, gain: float, broken: Set[Edge], joined: Set[Edge]) -> bool:
        """ Choose an edge to omit from the tour.
        tour: current tour to optimise
        t1: starting node for the current k-opt
        last: tail of the last edge added (t_2i-1)
        gain: current gain (curr_gain)
        broken: potential edges to remove (X)
        joined: potential edges to add (Y)
        return: whether we found an improved tour
        """
        if len(broken) == 4:
            pred, suc = tour.around(last)

            if self.matrix[pred][last] > self.matrix[suc][last]:
                around = [pred]
            else:
                around = [suc]
        else:
            around = tour.around(last)

        for t2i in around:
            xi = make_pair(last, t2i)
            curr_gain = gain + self.matrix[last][t2i]
            if xi not in joined and xi not in broken:
                added = joined.copy()
                removed = broken.copy()
                removed.add(xi)
                added.add(make_pair(t2i, t1))
                relink = curr_gain - self.matrix[t2i][t1]
                new_tour = tour.generate(removed, added)
                is_tour = False if len(new_tour) == 1 else True

                if not is_tour and len(added) > 2:
                    continue

                if get_hash(new_tour) in self.solutions:
                    return False

                if is_tour and relink > 0:
                    self.tour = new_tour
                    self.temp_length -= relink
                    return True
                else:
                    choice = self.choose_y(tour, t1, t2i, curr_gain, removed, joined)
                    if len(broken) == 2 and choice:
                        return True
                    else:
                        return choice
        return False

    def choose_y(self, tour: ListTour, t1: Node, t2i: Node, gain: float, broken: Set[Edge], joined: Set[Edge]):
        """ Choose an edge to add to the new tour.
        tour: current tour to optimise
        t1: starting node for the current k-opt
        t2i: tail of the last edge removed (t_2i)
        gain: current gain (Gi)
        broken: potential edges to remove (X)
        joined: potential edges to add (Y)
        return: whether we found an improved tour
        """
        ordered = self.closest(t2i, tour, gain, broken, joined)

        top = 5 if len(broken) == 2 else 1

        for node, (_, curr_gain) in ordered:
            yi = make_pair(t2i, node)
            added = joined.copy()
            added.add(yi)

            if self.choose_x(tour, t1, node, curr_gain, broken, added):
                return True

            top -= 1
            if top == 0:
                return False

        return False
