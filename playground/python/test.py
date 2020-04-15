import matplotlib.pyplot as plt

from src.algorithms.initial_tour import InitialTour
from src.algorithms.three_opt import ThreeOpt
from src.algorithms.two_opt import TwoOpt
from src.structures.matrix import Matrix
from src.tsp.generator import generator
from src.utils import draw_tour, get_length

tsp = [node for node in generator(200)]

weight_matrix = Matrix.weight_matrix(tsp)
init = InitialTour.greedy(weight_matrix)

two_opt = TwoOpt.optimize(init, weight_matrix)
three_opt = ThreeOpt.optimize(init, weight_matrix)
print(get_length(weight_matrix, init))
print(get_length(weight_matrix, two_opt))
print(get_length(weight_matrix, three_opt))
draw_tour(init, tsp, 'g')
draw_tour(two_opt, tsp, 'r')
plt.show()
draw_tour(init, tsp, 'g')
draw_tour(three_opt, tsp, 'b')
plt.show()
