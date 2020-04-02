from src.minimum_spanning_tree import MinimumSpanningTree
from src.tsp.ulysses16 import tsp
from src.weight_matrix import WeightMatrix
from src.subgradient_optimization import SubgradientOptimization

weight_matrix = WeightMatrix(tsp).matrix
mst = MinimumSpanningTree(weight_matrix)
print(mst.total_price)
print(mst.edges)

opt = SubgradientOptimization(weight_matrix)
