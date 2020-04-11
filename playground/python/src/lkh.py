from typing import List, Optional, Tuple

from src.algorithms.initial_tour import InitialTour
from src.algorithms.subgradient_optimization import SubgradientOptimization
from src.structures.graph import Graph, PoolEdges
from src.structures.matrix import Matrix
from src.structures.one_tree import OneTree
from src.structures.route import Route
from src.structures.solutions_set import SolutionSet


class LKH:
    nodes: List[Tuple[float, float]]  # вершины, вроде они не меняются по жизни, мб другой формат
    solutions_set: SolutionSet  # набор уже полученных решений
    weight_matrix: Matrix  # матрица весов
    selected_edges: PoolEdges  # уже выбранные ребра

    current_tour: Optional[Graph]  # Optional[ArrayListTree]  # текущее решение
    one_tree: Optional[OneTree]  # оптимальное дерево
    alpha_matrix: Optional[Matrix]  # матрица альфа близостей
    initial_generator: Optional[InitialTour]  # генератор начальных туров

    length: int

    def __init__(self, points: List[Tuple[float, float]]) -> None:
        self.length = len(points)
        self.nodes = points
        self.weight_matrix = Matrix.weight_matrix(points)
        self.solutions_set = SolutionSet()
        self.selected_edges = PoolEdges()

        self.current_tour = None
        self.one_tree = None
        self.alpha_matrix = None
        self.initial_generator = None

    def run(self) -> None:
        """ Пока тут просто шаблон нулевого запуска """
        self.__subgradient_optimization()
        # self.__one_tree()
        # self.__alpha_nearness()
        # self.__initial_tour()

    def __subgradient_optimization(self) -> None:
        opt = SubgradientOptimization(self.weight_matrix)  # переделать под WeightMatrix?
        # TODO: поменять текущую матрицу, уточнить этот вопрос

    def __one_tree(self) -> None:
        self.one_tree = OneTree.build(self.weight_matrix)  # node = 0

    def __alpha_nearness(self) -> None:
        self.alpha_matrix = Matrix.alpha_matrix(self.weight_matrix, self.one_tree)

    def __initial_tour(self) -> None:
        if self.initial_generator is None:
            self.initial_generator = InitialTour(self.alpha_matrix, self.weight_matrix, self.selected_edges)
        order = self.initial_generator.generate(self.solutions_set.get_best())
        tree = Route.build(self.nodes, order)
