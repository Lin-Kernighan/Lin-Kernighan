import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from src.algorithms.heuristics.lk_opt import LKOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import adjacency_matrix
from src.tsp.generator import generator

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

size = 1000

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

for _ in range(1):
    tsp = generator(size)
    matrix = adjacency_matrix(tsp)
    length, tour = InitialTour.greedy(matrix)

    two = LKOpt(length, tour, matrix, neighbours=25, bridge=False)
    two.optimize()
    print(two.length)
