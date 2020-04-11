from typing import List, Optional, Tuple

from src.algorithms.initial_tour import InitialTour
from src.algorithms.subgradient_optimization import SubgradientOptimization
from src.structures.matrix import Matrix
from src.structures.one_tree import OneTree
from src.structures.route import Route
from src.structures.solutions_set import SolutionSet


class LKH:
    nodes: List[Tuple[float, float]]  # вершины, вроде они не меняются по жизни, мб другой формат
    solutions_set: SolutionSet  # набор уже полученных решений
    weight_matrix: Matrix  # матрица весов

    current_tour: Optional[Route]  # текущее решение
    one_tree: Optional[OneTree]  # оптимальное дерево
    alpha_matrix: Optional[Matrix]  # матрица альфа близостей
    initial_generator: Optional[InitialTour]  # генератор начальных туров

    length: int

    def __init__(self, points: List[Tuple[float, float]]) -> None:
        self.length = len(points)
        self.nodes = points
        self.weight_matrix = Matrix.weight_matrix(points)
        self.solutions_set = SolutionSet()

        self.current_tour = None
        self.one_tree = None
        self.alpha_matrix = None
        self.initial_generator = None

    def run(self) -> None:
        """ Пока тут просто шаблон нулевого запуска """
        self.__subgradient_optimization()
        self.__one_tree()
        self.__alpha_nearness()
        self.__initial_tour()

    def __subgradient_optimization(self) -> None:
        opt = SubgradientOptimization.run(self.weight_matrix)  # ищем градиент
        SubgradientOptimization.make_move(opt.pi_sum, self.weight_matrix)  # сдвигаем матрицу к нужной

    def __one_tree(self) -> None:
        self.one_tree = OneTree.build(self.weight_matrix)  # node = 0

    def __alpha_nearness(self) -> None:
        self.alpha_matrix = Matrix.alpha_matrix(self.weight_matrix, self.one_tree)

    def __initial_tour(self) -> None:
        tour = InitialTour.helsgaun(self.alpha_matrix, self.solutions_set.get_best())
        self.current_tour = Route.build(self.nodes, tour)
