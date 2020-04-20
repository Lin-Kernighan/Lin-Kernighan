from math import log2

import matplotlib.pyplot as plt

from src.algorithms.heuristics.k_opt import KOpt
from src.algorithms.heuristics.three_opt import ThreeOpt
from src.algorithms.heuristics.two_opt import TwoOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import Matrix
from src.tsp.generator import generator
from src.utils import draw_plots_i_y, draw_plot_x_y

tsp = [node for node in generator(50)]

plt.style.use('ggplot')
matrix = Matrix.weight_matrix(tsp)
tour = InitialTour.greedy(matrix)
iterations = 25
swap = int(log2(len(tour)))

k_opt = KOpt(matrix, tour)
k_opt.optimize()

three_opt = ThreeOpt(tour, matrix)
three_tour = three_opt.optimize()

two_opt = TwoOpt(tour, matrix)
two_tour = two_opt.optimize()

k_frame = k_opt.collector.as_frame()
two_frame = two_opt.collector.as_frame()
three_frame = three_opt.collector.as_frame()

k_frame['time'] -= k_frame['time'][0]
two_frame['time'] -= two_frame['time'][0]
three_frame['time'] -= three_frame['time'][0]

plt.plot(k_frame['time'], k_frame['length'], label='k')
plt.plot(two_frame['time'], two_frame['length'], label='two')
plt.plot(three_frame['time'], three_frame['length'], label='three')
plt.xlabel('time')
plt.ylabel('length')
plt.legend()
plt.show()

draw_plots_i_y([k_frame, two_frame, three_frame], ['k', 'two', 'three'], ['delta', 'gain'], 'test')
draw_plot_x_y([k_frame, two_frame, three_frame], ['k', 'two', 'three'], 'time', 'length', 'test')
