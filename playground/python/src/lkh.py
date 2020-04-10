from typing import List, Optional, Tuple

from src.initial_tour import InitialTour
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
    initial_generator: Optional[InitialTour]  # генератор начальных туров

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
        self.initial_generator = None

    def run(self) -> None:
        """ Пока тут просто шаблон нулевого запуска """
        # self.__subgradient_optimization()
        # self.__one_tree()
        # self.__alpha_nearness()
        # self.__initial_tour()
        pass

    def __subgradient_optimization(self) -> None:
        opt = SubgradientOptimization(self.weight_matrix)  # переделать под WeightMatrix?
        # TODO: поменять текущую матрицу, уточнить этот вопрос

    def __one_tree(self) -> None:
        self.one_tree = OneTree(self.weight_matrix)  # node = 0

    def __alpha_nearness(self) -> None:
        self.alpha_matrix = AlphaMatrix(self.weight_matrix, self.one_tree)

    def __initial_tour(self) -> None:
        if self.initial_generator is None:
            self.initial_generator = InitialTour(self.alpha_matrix, self.weight_matrix, self.selected_edges)
        self.current_tour = self.initial_generator.generate(self.solutions_set.get_best())
