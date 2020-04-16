from src.algorithms.heuristics.three_opt import ThreeOpt
from src.algorithms.initial_tour import InitialTour
from src.algorithms.tabu_search import TabuSearch
from src.structures.matrix import Matrix
from src.structures.tabu_list import TabuDict
from src.tsp.generator import generator

tsp = [node for node in generator(70)]

matrix = Matrix.weight_matrix(tsp)
init = InitialTour.greedy(matrix)

tabu = TabuSearch(TabuDict(-1), ThreeOpt, init, matrix)
tabu.optimize(10, len(init) // 3)
print(tabu.best_result())
