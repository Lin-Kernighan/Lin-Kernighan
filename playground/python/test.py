from src.algorithms.heuristics.two_opt import TwoOpt
from src.algorithms.initial_tour import InitialTour
from src.algorithms.tabu_search import TabuSearch
from src.structures.matrix import Matrix
from src.structures.tabu_list import TabuDict
from src.tsp.generator import generator

tsp = [node for node in generator(500)]

matrix = Matrix.weight_matrix(tsp)
init = InitialTour.greedy(matrix)

tabu = TabuSearch(TabuDict(-1), TwoOpt, init, matrix)
tabu.optimize(100)
print(tabu.best_result())
