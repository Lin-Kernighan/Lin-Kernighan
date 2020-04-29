import warnings
from time import time

import matplotlib.pyplot as plt
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from src.algorithms.heuristics.k_opt import KOpt
from src.algorithms.heuristics.lkh_opt import LkhOpt
from src.algorithms.heuristics.three_opt import ThreeOpt
from src.algorithms.heuristics.two_opt import TwoOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import adjacency_matrix
from src.tsp.generator import generator
from src.utils import draw_tour, draw_plot_x_y

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

size = 500
tsp = generator(size)
matrix = adjacency_matrix(tsp)
l, t = InitialTour.greedy(matrix)
print(l)

t_start = time()
lkh = LkhOpt(matrix, True)
lkh.optimize()
print(time() - t_start)
print(lkh.length)
draw_tour(lkh.tour, tsp, 'r')
plt.show()

t_start = time()
lk = KOpt(l, t, matrix, True)
lk.optimize()
print(time() - t_start)
print(lk.length)
draw_tour(lk.tour, tsp, 'b')
plt.show()

t_start = time()
two = TwoOpt(l, t, matrix)
two.optimize()
print(time() - t_start)
print(two.length)
draw_tour(two.tour, tsp, 'g')
plt.show()

t_start = time()
three = ThreeOpt(l, t, matrix)
three.optimize()
print(time() - t_start)
print(three.length)
draw_tour(three.tour, tsp, 'y')
plt.show()

a = lkh.collector.as_frame()
b = lk.collector.as_frame()
c = two.collector.as_frame()
d = three.collector.as_frame()
a['time'] -= a['time'][0]
b['time'] -= b['time'][0]
c['time'] -= c['time'][0]
d['time'] -= d['time'][0]

draw_plot_x_y([a, b, c, d], ['lkh', 'lk', 'two', 'three'], 'time', 'length', 'plot', 'src')
