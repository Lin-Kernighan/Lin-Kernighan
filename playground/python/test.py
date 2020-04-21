from math import log2

from src.algorithms.heuristics.k_opt import KOpt
from src.algorithms.heuristics.three_opt import ThreeOpt
from src.algorithms.heuristics.two_opt import TwoOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import Matrix
from src.test import save_test
from src.tsp.generator import generator

tsp = [node for node in generator(50)]

matrix = Matrix.weight_matrix(tsp)
tour = InitialTour.greedy(matrix)
iterations = 15
swap = int(log2(len(tour)))

k_opt = KOpt(tour, matrix)
three_opt = ThreeOpt(tour, matrix)
two_opt = TwoOpt(tour, matrix)

save_test([KOpt, TwoOpt, ThreeOpt], ['k_opt', 'two_opt', 'three_opt'], 50)
