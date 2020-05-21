import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from src.algorithms.heuristics.three_opt import ThreeOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import adjacency_matrix
from src.tsp.generator import generator
from src.utils import get_length

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

size = 100

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

tsp = generator(size)
matrix = adjacency_matrix(tsp)
length, tour = InitialTour.greedy(matrix)

two = ThreeOpt(length, tour, matrix)
two.optimize()

print(get_length(matrix, two.tour))
print(two.length)
