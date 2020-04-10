from random import randrange
from typing import List, Set, Optional, Tuple

from src.structures.alpha_matrix import AlphaMatrix
from src.structures.graph import Graph
from src.structures.one_tree import OneTree
from src.structures.solutions_set import SolutionSet
from src.structures.weight_matrix import WeightMatrix
from src.subgradient_optimization import SubgradientOptimization


class LKH:
    nodes: List[Tuple[float, float]]  # вершины, вроде они не меняются по жизни, мб другой формат
    solutions_set: SolutionSet  # набор уже полученных решений
    weight_matrix: WeightMatrix  # матрица весов

    current_graph: Optional[Graph]  # текущее решение
    one_tree: Optional[OneTree]  # оптимальное дерево
    alpha_matrix: Optional[AlphaMatrix]  # матрица альфа близостей
    # not_selected_edges: Optional[Heap]  # доступные не выбранные ранее ребра
    selected_edges: Set[Tuple[int, int]]  # уже выбранные ребра

    length: int

    def __init__(self, points: List[Tuple[float, float]]) -> None:
        self.length = len(points)
        self.nodes = points
        self.weight_matrix = WeightMatrix(points)
        self.solutions_set = SolutionSet()

        self.current_graph = None
        self.one_tree = None
        self.alpha_matrix = None
        # self.not_selected_edges = None
        # self.selected_edges = None

    def __subgradient_optimization(self) -> None:
        opt = SubgradientOptimization(self.weight_matrix.matrix)  # переделать под WeightMatrix?
        # TODO: поменять текущую матрицу, уточнить этот вопрос

    def __one_tree(self) -> None:
        self.one_tree = OneTree(self.weight_matrix.matrix)  # node = 0

    def __alpha_nearness(self) -> None:
        self.alpha_matrix = AlphaMatrix(self.weight_matrix.matrix, self.one_tree)

    def __initial_tour(self) -> None:
        best_solution: Optional[Graph] = self.solutions_set.get_best()
        first_node = current_node = randrange(0, self.length - 1)  # пункт первый "choose a random node i"
        