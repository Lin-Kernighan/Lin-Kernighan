from math import log2

from src.algorithms.heuristics.k_opt import KOpt
from src.algorithms.heuristics.three_opt import ThreeOpt
from src.algorithms.heuristics.two_opt import TwoOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import Matrix
from src.tsp.generator import generator
from src.utils import draw_plots

tsp = [node for node in generator(50)]

# plt.style.use('ggplot')
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

draw_plots([
    k_opt.collector.as_frame(),
    two_opt.collector.as_frame(),
    three_opt.collector.as_frame()
],
    ['k_opt', 'two_opt', 'three_opt'],
    ['length', 'gain', 'time'],
    'test_250_3'
)
