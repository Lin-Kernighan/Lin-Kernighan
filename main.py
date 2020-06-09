import faulthandler
import logging
from time import time

from lin_kernighan.algorithms.structures.matrix import adjacency_matrix
from lin_kernighan.algorithms.utils.generator import generator
from lin_kernighan.lkh_search import LKHSearch

if __name__ == '__main__':
    size = 500
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    faulthandler.enable()
    full = 0.0
    num = 1
    t_start = time()
    for _ in range(num):
        tsp = generator(size)
        matrix = adjacency_matrix(tsp)

        t_start = time()
        opt = LKHSearch(matrix)
        opt.optimize()
        full += (time() - t_start)

    logging.info(f'end: {full / num}')
