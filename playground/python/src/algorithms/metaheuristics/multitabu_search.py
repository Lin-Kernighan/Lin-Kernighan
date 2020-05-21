# import numpy as np
#
# from src.structures.collector import Collector
# from src.structures.tabu_list import TabuSet
#
#
# class MultiTabuSearch:
#
#     def __init__(self, optimization, tour: np.ndarray, matrix: np.ndarray):
#         self.collector = Collector(['length', 'gain'], {'tabu search': len(tour), 'type': optimization.__name__})
#         self.data = TabuSet()
#         self.tour = tour
#         self.matrix = matrix
#         self.local_optimization = optimization
#
