import warnings
from time import time

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from src.algorithms.heuristics.lkh_opt import LkhOpt
from src.structures.matrix import adjacency_matrix
from src.tsp.generator import generator

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

size = 200

x = 0
for _ in range(25):
    tsp = generator(size)
    matrix = adjacency_matrix(tsp)
    t_start = time()
    lkh = LkhOpt(matrix, False)
    lkh.optimize()
    x += (time() - t_start)
print(x / 25)

# t_start = time()
# lk = KOpt(l, t, matrix, True)
# lk.optimize()
# print(time() - t_start)
# print(lk.length)
# draw_tour(lk.tour, tsp, 'b')
# plt.show()

# a = lkh.collector.as_frame()
# b = lk.collector.as_frame()
# a['time'] -= a['time'][0]
# b['time'] -= b['time'][0]

# draw_plot_x_y([a, b], ['lkh', 'lk'], 'time', 'length', 'plot', 'src')
