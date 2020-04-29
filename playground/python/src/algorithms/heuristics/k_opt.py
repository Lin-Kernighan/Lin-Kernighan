from typing import Tuple, Set, Dict

from numpy import ndarray, zeros

from src.algorithms.heuristics.abc_opt import AbcOpt
from src.structures.collector import Collector
from src.structures.tabu_list import TabuSet
from src.structures.tour.list_tour import ListTour
from src.utils import make_pair

Edge = Tuple[int, int]
Node = int


class KOpt(AbcOpt):

    def __init__(self, length: float, tour: ndarray, matrix: ndarray, dlb=True):
        super().__init__(length, tour, matrix)
        self.solutions: Set[str] = set()
        self.neighbours: Dict[Node, list] = dict()
        self.dlb = zeros(self.size, dtype=bool) if dlb else None
        self.temp_length = self.length

    def optimize(self) -> ndarray:
        """ Global loop which restarts at each improving solution. """
        self.calc_neighbours()

        iteration, self.collector = 0, Collector(['length', 'gain'], {'k_opt': self.size})
        self.collector.update({'length': self.length, 'gain': 0})

        better = True
        while better:  # Restart the loop each time we find an improving candidate
            print(f'{iteration} : {self.length}')
            better = self.improve()
            gain = self.length - self.temp_length
            self.length = self.temp_length
            self.collector.update({'length': self.length, 'gain': gain})
            # Paths always begin at 0 so this should manage to find duplicate solutions
            self.solutions.add(str(self.tour))
            iteration += 1

        return self.tour

    def tabu_optimize(self, tabu_list: TabuSet, collector: Collector) -> ndarray:
        """ Запуск эвристики под управление tabu search """
        self.calc_neighbours()
        self.solutions = self.solutions | tabu_list.data
        self.collector = collector  # закинули существующие решения
        self.collector.update({'length': self.length, 'gain': 0})

        better = True
        while better:
            better = self.improve()
            gain = self.length - self.temp_length
            self.length = self.temp_length
            self.collector.update({'length': self.length, 'gain': gain})
            tabu_list.append(self.tour, self.length)
            self.solutions.add(str(self.tour))

        return self.tour

    def calc_neighbours(self) -> None:
        """ Просто собираем всех соседей """
        for i in self.tour:
            self.neighbours[i] = []
            for j, dist in enumerate(self.matrix[i]):
                if dist > 0 and j in self.tour:
                    self.neighbours[i].append(j)  # dict(i: [j1, j2, j3...])

    def improve(self):
        """ Start the LK algorithm with the current tour. """
        tour = ListTour(self.tour)

        # Find all valid 2-opt moves and try them
        for index in range(len(tour)):
            t1 = tour[index]
            if self.dlb is not None \
                    and self.dlb[t1] and self.dlb[(t1 - 1) % self.size] and self.dlb[(t1 + 1) % self.size]:
                continue

            around = tour.around(t1)  # вытащили двух соседей

            for t2 in around:  # для каждой соседней вершины
                broken = {make_pair(t1, t2)}  # ломаем ребро
                # Initial savings
                gain = self.matrix[t1][t2]  # которое стоило столько
                close = self.closest(t2, tour, gain, broken, set())  # вернули потенциальных клиентов
                tries = 5  # Number of neighbours to try, ну это от LK

                for t3, (_, curr_gain) in close:  # и вот для этих соседей
                    # Make sure that the new node is none of t_1's neighbours
                    # so it does not belong to the tour.
                    if t3 in around:
                        continue

                    joined = {make_pair(t2, t3)}  # пытаемся соединить t2, t3

                    # The positive curr_gain is taken care of by `closest()`
                    if self.choose_x(tour, t1, t3, curr_gain, broken, joined):
                        # Return to Step 2, that is the initial loop

                        if self.dlb is not None:
                            for x, y in broken:
                                self.dlb[x] = self.dlb[y] = False
                            for x, y in joined:
                                self.dlb[x] = self.dlb[y] = False

                        return True
                    # Else try the other options

                    tries -= 1
                    # Explored enough nodes, change t_2
                    if tries == 0:
                        break

            if self.dlb is not None:
                self.dlb[t1] = True

        return False

    def closest(self, t2i: Node, tour: ListTour, gain: float, broken: Set[Edge], joined: Set[Edge]) -> list:
        """
        Find the closest neighbours of a node ordered by potential gain.  As a
        side-effect, also compute the partial improvement of joining a node.
        Parameters:
            - t2i: node to relink from
            - tour: current tour to optimise
            - gain: current gain
            - broken: set of edges to remove (X)
            - joined: set of edges to join (Y)
        Return: sorted list of neighbours based on potential improvement with
        next omission
        """
        neighbours = {}  # очередной словарь соседей

        # Create the neighbours of t_2i
        for node in self.neighbours[t2i]:  # и вот для того t2 берем всех соседей
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
                new_tour = tour.generate(removed, added)
                is_tour = False if len(new_tour) == 1 else True

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
