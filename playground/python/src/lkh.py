from typing import List, Optional, Tuple

from src.structures.alpha_matrix import AlphaMatrix
from src.structures.graph import Graph, PoolEdges
from src.structures.one_tree import OneTree
from src.structures.solutions_set import SolutionSet
from src.structures.weight_matrix import WeightMatrix
from src.subgradient_optimization import SubgradientOptimization


class LKH:
    nodes: List[Tuple[float, float]]  # вершины, вроде они не меняются по жизни, мб другой формат
    solutions_set: SolutionSet  # набор уже полученных решений
    weight_matrix: WeightMatrix  # матрица весов
    selected_edges: PoolEdges  # уже выбранные ребра

    current_tour: Optional[Graph]  # текущее решение
    one_tree: Optional[OneTree]  # оптимальное дерево
    alpha_matrix: Optional[AlphaMatrix]  # матрица альфа близостей
    # not_selected_edges: Optional[Heap]  # доступные не выбранные ранее ребра

    length: int

    def __init__(self, points: List[Tuple[float, float]]) -> None:
        self.length = len(points)
        self.nodes = points
        self.weight_matrix = WeightMatrix(points)
        self.solutions_set = SolutionSet()
        self.selected_edges = PoolEdges()

        self.current_tour = None
        self.one_tree = None
        self.alpha_matrix = None
        # self.not_selected_edges = None

    def run(self) -> None:
        # self.__subgradient_optimization()
        # self.__one_tree()
        # self.__alpha_nearness()
        # self.__initial_tour()
        pass

    def __subgradient_optimization(self) -> None:
        opt = SubgradientOptimization(self.weight_matrix.matrix)  # переделать под WeightMatrix?
        # TODO: поменять текущую матрицу, уточнить этот вопрос

    def __one_tree(self) -> None:
        self.one_tree = OneTree(self.weight_matrix.matrix)  # node = 0

    def __alpha_nearness(self) -> None:
        self.alpha_matrix = AlphaMatrix(self.weight_matrix.matrix, self.one_tree)

    def __initial_tour(self) -> None:
        self.current_tour = InitialTour()
