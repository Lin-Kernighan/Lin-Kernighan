# from src.algorithms.heuristics.k_opt import KOpt
# from src.algorithms.heuristics.three_opt import ThreeOpt
# from src.algorithms.heuristics.two_opt import TwoOpt
# from src.test import save_test
#
# save_test([TwoOpt, ThreeOpt], ['two_opt', 'three_opt'], 500)

from src.structures.matrix import adjacency_matrix, betta_matrix, alpha_matrix
from src.tsp.generator import generator
from src.utils import print_matrix

tsp = generator(5)
matrix = adjacency_matrix(tsp)

# length, src, dst = one_tree(matrix)
# print(length)
# print(np.dstack((src, dst)))
#
# tree = OneTree.build(matrix)
# print(tree.edges)
# print(tree.total_price)
#
# topology = one_tree_topology(matrix)
# print(topology)

print_matrix(matrix)
b = betta_matrix(matrix)
a = alpha_matrix(matrix)

print_matrix(a - b)
