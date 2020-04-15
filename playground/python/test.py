import matplotlib.pyplot as plt

from src.algorithms.initial_tour import InitialTour
from src.algorithms.k_opt import KOpt
from src.structures.matrix import Matrix
from src.tsp.generator import generator
from src.utils import draw_tour, get_length

tsp = [node for node in generator(100)]

weight_matrix = Matrix.weight_matrix(tsp)
init = InitialTour.greedy(weight_matrix)

k_opt = KOpt(tsp)
draw_tour(k_opt.tour, tsp, 'r')
print(get_length(k_opt.matrix, k_opt.tour))
k_opt.optimize()
draw_tour(k_opt.tour, tsp, 'b')
print(get_length(k_opt.matrix, k_opt.tour))
plt.show()
