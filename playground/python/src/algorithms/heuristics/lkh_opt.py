# from typing import List, Tuple
#
# from src.algorithms.heuristics.k_opt import KOpt
# from src.structures.matrix import Matrix
# from src.structures.one_tree import OneTree
#
# Edge = Tuple[int, int]
# Node = int
#
#
# class LkhOpt(KOpt):
#
#     def __init__(self, tour: List[Node], adjacency: Matrix, excess: float = None):
#         super().__init__(tour, adjacency)
#         opt = SubgradientOptimization.run(self.matrix)  # ищем градиент
#         SubgradientOptimization.make_move(opt.pi_sum, self.matrix)  # сдвигаем матрицу к нужной, проверить
#         self.one_tree = OneTree.build(self.matrix)
#         self.alpha_matrix = Matrix.alpha_matrix(self.matrix, self.one_tree)
#         self.excess = excess if excess is not None else 1 / self.length * self.one_tree.total_price
#         if not tour:
#             tour = InitialTour.helsgaun(self.alpha_matrix, self.solutions_set.get_best(), excess)
#
#     def calc_neighbours(self) -> None:
#         for i in self.tour:
#             self.neighbours[i] = []
#             for j, nearness in enumerate(self.alpha_matrix[i]):
#                 if i == j:
#                     continue
#                 if nearness > self.excess and j in self.tour:
#                     self.neighbours[i].append(j)
