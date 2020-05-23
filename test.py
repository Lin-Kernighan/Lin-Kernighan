import faulthandler
import logging

from src.algorithms.lk_opt import LKOpt
from src.algorithms.structures.matrix import adjacency_matrix
from src.algorithms.two_opt import TwoOpt
from src.algorithms.utils.initial_tour import InitialTour
from src.tsp.generator import generator

size = 1500
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

faulthandler.enable()
full = 0.0
num = 1
for _ in range(num):
    tsp = generator(size)
    matrix = adjacency_matrix(tsp)
    length, tour = InitialTour.greedy(matrix)

    lk = LKOpt(length, tour.copy(), matrix.copy(), neighbours=5, k=5, dlb=True)
    lk.optimize()

    # two = TwoOpt(length, tour.copy(), matrix.copy(), mul=15)
    # two.optimize()
    #
    # full += (two.length - lk.length)
#
#     # three = ThreeOpt(length, tour.copy(), matrix.copy())
#     # three.optimize()
#
#     two = TwoOpt(length, tour.copy(), matrix.copy())
#     two.optimize()
#
#     lk_frame = lk.collector.as_frame()
#     # lkh_frame = lkh.collector.as_frame()
#     # three_frame = three.collector.as_frame()
#     two_frame = two.collector.as_frame()
#
#     draw_plot_x_y(
#         [lk_frame, two_frame],
#         ['lk', 'two'],
#         'time', 'length',
#         None,
#         None
#     )
#
logging.info(f'end: {full / num}')
