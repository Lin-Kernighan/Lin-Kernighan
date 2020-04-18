from math import log2

import matplotlib.pyplot as plt

from src.algorithms.heuristics.two_opt import TwoOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import Matrix
from src.tsp.generator import generator
from src.utils import draw_tour

tsp = [node for node in generator(500)]

matrix = Matrix.weight_matrix(tsp)
tour = InitialTour.greedy(matrix)
iterations = 25
swap = int(log2(len(tour)))

# tabu = TabuSearch(TabuDict(-1), ThreeOpt, tour, matrix)
# tabu.optimize(iterations, swap)
# print(tabu.best_result())
# draw_tour(tabu.best_tour(), tsp, 'r')

opt = TwoOpt(tour, matrix)
tak = opt.optimize()
print(opt.collector)
draw_tour(tak, tsp, 'r')
plt.show()

# tabu = TabuSearch(TabuDict(-1), TwoOpt, tour, matrix)
# tabu.optimize(iterations, swap)
# draw_tour(tabu.best_tour(), tsp, 'b')
# plt.show()

# opt = KOpt(matrix, tabu.best_tour())
# opt.optimize()
# draw_tour(opt.tour, tsp, 'g')
