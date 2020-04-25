# from src.algorithms.heuristics.k_opt import KOpt
# from src.algorithms.heuristics.three_opt import ThreeOpt
# from src.algorithms.heuristics.two_opt import TwoOpt
# from src.test import save_test
#
# save_test([TwoOpt, ThreeOpt], ['two_opt', 'three_opt'], 500)
import pickle
from math import isclose

from src.structures.matrix import adjacency_matrix, betta_matrix, alpha_matrix
from src.tsp.generator import generator
from src.utils import print_matrix


def main():
    tsp = generator(30)
    # with open('breaking.pkl', 'rb') as f:
    #     tsp = pickle.load(f)

    matrix = adjacency_matrix(tsp)

    print_matrix(matrix)
    b = betta_matrix(matrix)
    a = alpha_matrix(matrix)

    new_matr = (a - b)[1:, 1:]
    if not isclose(new_matr.sum(), 0, abs_tol=0.01):
        with open('breaking.pkl', 'wb') as f:
            pickle.dump(tsp, f)
        print('-' * 100)
        print_matrix(a - b)
        exit()


if __name__ == '__main__':
    for i in range(10000):
        main()
