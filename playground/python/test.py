from math import log2

from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import Matrix
from src.tsp.generator import generator

tsp = [node for node in generator(20)]

matrix = Matrix.weight_matrix(tsp)
tour = InitialTour.greedy(matrix)
iterations = 25
swap = int(log2(len(tour)))

# tabu = TabuSearch(TabuDict(-1), ThreeOpt, tour, matrix)
# tabu.optimize(iterations, swap)
# print(tabu.best_result())
# draw_tour(tabu.best_tour(), tsp, 'r')

# tak = TwoOpt(tour, matrix).optimize()
# draw_tour(tak, tsp, 'r')
# plt.show()

from src.tsp.tsp_loader import TspLoader

print(TspLoader.tsplib_deserializer('http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/d657.tsp'))

# tabu = TabuSearch(TabuDict(-1), TwoOpt, tour, matrix)
# tabu.optimize(iterations, swap)
# draw_tour(tabu.best_tour(), tsp, 'b')
# plt.show()

# opt = KOpt(matrix, tabu.best_tour())
# opt.optimize()
# draw_tour(opt.tour, tsp, 'g')
