from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict

from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import Matrix
from src.utils import make_pair, get_length

Edge = Tuple[int, int]
Node = int


@dataclass
class Tour:
    tour: List[Node]
    size: int = field(init=False)
    edges: Set[Edge] = field(init=False)

    def __post_init__(self):
        self.size = len(self.tour)
        self.edges = set()
        for i in range(self.size):
            self.edges.add(make_pair(self.tour[i - 1], self.tour[i]))

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> Node:
        """ Вершина по номеру в туре """
        return self.tour[index % self.size]

    def __contains__(self, edge: Edge) -> bool:
        """ Наличие ребра в туре """
        return edge in self.edges

    def index(self, node: Node) -> int:
        """ Номер вершины в туре """
        return self.tour.index(node)

    def around(self, node: int) -> Tuple[Node, Node]:
        """ Предыдущая вершина и следующая текущей веришны """
        index = self.index(node)
        return self[index - 1], self[index + 1]

    def successor(self, index: int) -> Node:
        """ Следующий """
        return self[index + 1]

    def predecessor(self, index: int) -> Node:
        """ Предыдущий """
        return self[index - 1]

    def generate(self, broken: Set[Edge], joined: Set[Edge]) -> Tuple[bool, list]:
        """ Немного магии
        Создаем новый тур, а потом проверяем его на целостность и наличие циклов
        """
        # New edges: old edges minus broken, plus joined
        edges = (self.edges - broken) | joined

        # If we do not have enough edges, we cannot form a tour -- should not
        if len(edges) < self.size:
            return False, []

        successors = {}
        node = 0

        # Build the list of successors
        while len(edges) > 0:
            i = j = 0
            for i, j in edges:
                if i == node:
                    successors[node] = j
                    node = j
                    break
                elif j == node:
                    successors[node] = i
                    node = i
                    break

            edges.remove((i, j))

        # Similarly, if not every node has a successor, this can not work
        if len(successors) < self.size:
            return False, []

        successor = successors[0]
        new_tour = [0]
        visited = set(new_tour)

        # If we already encountered a node it means we have a loop
        while successor not in visited:
            visited.add(successor)
            new_tour.append(successor)
            successor = successors[successor]

        # If we visited all nodes without a loop we have a tour
        return len(new_tour) == self.size, new_tour


class KOpt:

    def __init__(self, points: List[Tuple[float, float]]):
        self.matrix: Matrix = Matrix.weight_matrix(points)
        self.tour: List[Node] = InitialTour.greedy(self.matrix)
        self.solutions: Set[str] = set()
        self.neighbours: Dict[Node, List[Node]] = dict()

    def optimize(self) -> None:
        """ Global loop which restarts at each improving solution. """
        better = True

        for i in self.tour:  # просто собираем всех соседей
            self.neighbours[i] = []

            for j, dist in enumerate(self.matrix[i]):
                if dist > 0 and j in self.tour:
                    self.neighbours[i].append(j)  # dict(i: [j1, j2, j3...])

        iteration = 0
        while better:  # Restart the loop each time we find an improving candidate
            print(f'{iteration} : {get_length(self.matrix, self.tour)}')
            better = self.improve()
            # Paths always begin at 0 so this should manage to find duplicate solutions
            self.solutions.add(str(self.tour))  # не уверен, что он прав
            iteration += 1
        print('done')

    def improve(self):
        """ Start the LK algorithm with the current tour. """
        tour = Tour(self.tour)

        # Find all valid 2-opt moves and try them
        for index in range(len(tour)):  # сделано так, чтобы индекс не переполнялся, ох уж этот питон
            t1 = tour[index]
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
                        return True
                    # Else try the other options

                    tries -= 1
                    # Explored enough nodes, change t_2
                    if tries == 0:
                        break

        return False

    def closest(self, t2i: Node, tour: Tour, gain: float, broken: Set[Edge], joined: Set[Edge]) -> list:
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

                # TODO verify it is enough, but we do check if the tour is
                # valid first thing in `choose_x` so this should be sufficient, что он имеет ввиду?
                # Check that "x_i+1 exists"  # вообще оно всегда существует, если я верно понял
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

    def choose_x(self, tour: Tour, t1: Node, last: Node, gain: float, broken: Set[Edge], joined: Set[Edge]) -> bool:
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

            # Give priority to the longest edge for x_4
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
                added = deepcopy(joined)
                removed = deepcopy(broken)

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

    def choose_y(self, tour, t1, t2i, gain, broken, joined):
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

        for node, (_, Gi) in ordered:
            yi = make_pair(t2i, node)
            added = deepcopy(joined)
            added.add(yi)

            # Stop at the first improving tour
            if self.choose_x(tour, t1, node, Gi, broken, added):
                return True

            top -= 1
            # Tried enough options
            if top == 0:
                return False

        return False
