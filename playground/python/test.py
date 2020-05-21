import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from src.algorithms.heuristics.k import LKOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import adjacency_matrix
from src.tsp.generator import generator

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

size = 500

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

tsp = generator(size)
matrix = adjacency_matrix(tsp)
length, tour = InitialTour.greedy(matrix)

two = LKOpt(length, tour, matrix, radius=20)
two.optimize()
print(two.length)

# import numpy as np
#
# print(np.zeros([4, 10], dtype=int))
