from math import log2

from src.algorithms.heuristics.k_opt import KOpt
from src.algorithms.heuristics.three_opt import ThreeOpt
from src.algorithms.heuristics.two_opt import TwoOpt
from src.algorithms.initial_tour import InitialTour
from src.algorithms.tabu_search import TabuSearch
from src.structures.matrix import Matrix
from src.structures.tabu_list import TabuSet
from src.tsp.generator import generator
from src.utils import draw_plots_i_y, draw_plot_x_y

tsp = [node for node in generator(50)]

matrix = Matrix.weight_matrix(tsp)
tour = InitialTour.greedy(matrix)
iterations = 15
swap = int(log2(len(tour)))

k_opt = TabuSearch(TabuSet(-1), KOpt, tour, matrix)
k_opt.optimize(iterations, swap)
print(k_opt.collector)

three_opt = TabuSearch(TabuSet(-1), ThreeOpt, tour, matrix)
three_opt.optimize(iterations, swap)

two_opt = TabuSearch(TabuSet(-1), TwoOpt, tour, matrix)
two_opt.optimize(iterations, swap)

k_frame = k_opt.collector.as_frame()
two_frame = two_opt.collector.as_frame()
three_frame = three_opt.collector.as_frame()

k_frame['time'] -= k_frame['time'][0]
two_frame['time'] -= two_frame['time'][0]
three_frame['time'] -= three_frame['time'][0]

draw_plots_i_y([k_frame, two_frame, three_frame], ['k', 'two', 'three'], ['delta', 'gain'], 'test')
draw_plot_x_y([k_frame, two_frame, three_frame], ['k', 'two', 'three'], 'time', 'length', 'test')
