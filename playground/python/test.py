import matplotlib.pyplot as plt

from src.algorithms.initial_tour import InitialTour
from src.algorithms.two_opt import TwoOpt
from src.structures.matrix import Matrix
from src.tsp.generator import generator
from src.utils import draw_tour

tsp = [node for node in generator(100)]

weight_matrix = Matrix.weight_matrix(tsp)
init = InitialTour.greedy(weight_matrix)
opt = TwoOpt.run(weight_matrix, init)
draw_tour(init, tsp, 'g')
draw_tour(opt, tsp, 'r')
plt.show()
