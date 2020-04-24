# from src.algorithms.heuristics.k_opt import KOpt
# from src.algorithms.heuristics.three_opt import ThreeOpt
# from src.algorithms.heuristics.two_opt import TwoOpt
# from src.test import save_test
#
# save_test([TwoOpt, ThreeOpt], ['two_opt', 'three_opt'], 500)

from src.structures.matrix import adjacency_matrix, betta_matrix
from src.tsp.generator import generator

tsp = generator(10)
matrix = adjacency_matrix(tsp)

betta_matrix(matrix)
# alpha_matrix(matrix)
