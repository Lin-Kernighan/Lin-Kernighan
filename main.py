import faulthandler
import logging
from time import time

from src.algorithms.structures.matrix import adjacency_matrix
from src.algorithms.utils.initial_tour import greedy
from src.tabu_search import TabuSearch
from src.algorithms.utils.generator import generator

if __name__ == '__main__':
    size = 200
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    faulthandler.enable()
    full = 0.0
    num = 1
    t_start = time()
    for _ in range(num):
        tsp = generator(size)
        matrix = adjacency_matrix(tsp)
        length, tour = greedy(matrix)

        t_start = time()
        opt = TabuSearch('lk_opt', matrix)
        opt.optimize()
        full += (time() - t_start)

    logging.info(f'end: {full / num}')
