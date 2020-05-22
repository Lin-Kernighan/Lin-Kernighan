import logging
from time import time

from src.algorithms.lk_opt import LKOpt
from src.algorithms.structures.matrix import adjacency_matrix
from src.algorithms.three_opt import ThreeOpt
from src.algorithms.two_opt import TwoOpt
from src.algorithms.utils.initial_tour import InitialTour
from src.tsp.generator import generator

size = 300
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

full = 0.0
num = 1
for _ in range(num):
    tsp = generator(size)
    matrix = adjacency_matrix(tsp)
    length, tour = InitialTour.greedy(matrix)

    t_start = time()
    two = LKOpt(length, tour.copy(), matrix.copy())
    two.optimize()
    t_end = time() - t_start
    logging.info(f'time: {t_end}')
    logging.info(f'length: {two.length}')
    full += t_end

    # t_start = time()
    # two = LKOpt(length, tour.copy(), matrix.copy())
    # two.optimize()
    # t_end = time() - t_start
    # logging.info(f'time: {t_end}')
    # logging.info(f'length: {two.length}')

logging.info(f'full: {full / num}')
