import logging
from time import time

from src.algorithms.heuristics.lk_opt import LKOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import adjacency_matrix
from src.tsp.generator import generator

size = 200
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

for _ in range(1):
    t_start = time()
    tsp = generator(size)
    matrix = adjacency_matrix(tsp)
    length, tour = InitialTour.greedy(matrix)

    two = LKOpt(length, tour, matrix, neighbours=5, bridge=True)
    two.optimize()
    logging.info(f'time: {time() - t_start}')
    logging.info(f'length: {two.length}')
