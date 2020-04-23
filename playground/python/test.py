# from src.algorithms.heuristics.k_opt import KOpt
# from src.algorithms.heuristics.three_opt import ThreeOpt
# from src.algorithms.heuristics.two_opt import TwoOpt
# from src.test import save_test
#
# save_test([KOpt, TwoOpt, ThreeOpt], ['k_opt', 'two_opt', 'three_opt'], 500)

from time import time

from src.structures.matrix import Matrix
from src.tsp.generator import generator

tsp = [node for node in generator(100)]
matrix = Matrix.adjacency_matrix(tsp)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

X = csr_matrix(matrix)
t_start = time()
# noinspection PyTypeChecker
Tcsr: csr_matrix = minimum_spanning_tree(X)
print(time() - t_start)
print(type(Tcsr.tocoo().col))
print(type(Tcsr.tocoo().row))

# t_start = time()
# one = OneTree.build(matrix, 0)
# print(time() - t_start)


# t_start = time()
# opt = SubgradientOptimization.run(matrix)
# print(time() - t_start)
