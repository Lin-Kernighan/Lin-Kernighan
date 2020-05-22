import logging
from time import time

from src.algorithms.heuristics.lkh_opt import LKHOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import adjacency_matrix
from src.tsp.generator import generator

size = 1000
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

full = 0.0
for _ in range(1):
    tsp = generator(size)
    matrix = adjacency_matrix(tsp)
    length, tour = InitialTour.greedy(matrix)

    t_start = time()
    two = LKHOpt(length, tour.copy(), matrix.copy())
    two.optimize()
    t_end = time() - t_start
    logging.info(f'time: {t_end}')
    logging.info(f'length: {two.length}')

    # t_start = time()
    # two = LKOpt(length, tour.copy(), matrix.copy())
    # two.optimize()
    # t_end = time() - t_start
    # logging.info(f'time: {t_end}')
    # logging.info(f'length: {two.length}')

logging.info(f'full: {full}')
