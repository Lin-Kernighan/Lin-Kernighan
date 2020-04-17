from math import log10

from src.algorithms.heuristics.three_opt import ThreeOpt
from src.algorithms.heuristics.two_opt import TwoOpt
from src.algorithms.initial_tour import InitialTour
from src.algorithms.tabu_search import TabuSearch
from src.structures.matrix import Matrix
from src.structures.tabu_list import TabuDict
from src.tsp.generator import generator

tsp = [node for node in generator(100)]

matrix = Matrix.weight_matrix(tsp)
tour = InitialTour.greedy(matrix)
iterations = 15
swap = int(log10(len(tour)))

tabu = TabuSearch(TabuDict(-1), ThreeOpt, tour, matrix)
tabu.optimize(iterations, swap)
print(tabu.best_result())

tabu = TabuSearch(TabuDict(-1), TwoOpt, tour, matrix)
tabu.optimize(iterations, swap)
print(tabu.best_result())
