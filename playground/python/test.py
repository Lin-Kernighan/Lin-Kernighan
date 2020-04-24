# from src.algorithms.heuristics.k_opt import KOpt
# from src.algorithms.heuristics.three_opt import ThreeOpt
# from src.algorithms.heuristics.two_opt import TwoOpt
# from src.test import save_test
#
# save_test([KOpt, TwoOpt, ThreeOpt], ['k_opt', 'two_opt', 'three_opt'], 500)
from time import time

from src.algorithms.subgradient_optimization import run
from src.structures.matrix import Matrix
from src.tsp.generator import generator

tsp = [node for node in generator(1000)]
matrix = Matrix.adjacency_matrix(tsp)
# import numba
#
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import minimum_spanning_tree
# #
# X = csr_matrix(matrix)
# # #
# Tcsr: csr_matrix = minimum_spanning_tree(X)
# coo = Tcsr.tocoo()
# first = coo.col
# second = coo.row
# print(first)
# print(second)
# temp = np.dstack(first, second)
# for x, y in temp:
#     print(x, y)

t_start = time()
run(matrix)
print(time() - t_start)
