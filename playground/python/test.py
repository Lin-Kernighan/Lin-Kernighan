from typing import List

import matplotlib.pyplot as plt

from src.graph import Edge
from src.one_tree import OneTree
from src.subgradient_optimization import SubgradientOptimization
from src.tsp.oliver30 import tsp
from src.weight_matrix import WeightMatrix


def draw(edges: List[Edge], nodes: List[List[float]]):
    for edge in edges:
        [x1, y1] = nodes[edge.dst]
        [x2, y2] = nodes[edge.src]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color='r')

    for idx, node in enumerate(nodes):
        plt.annotate(f'{idx}:({node[0]},{node[1]})', node, size=9)
    plt.show()


weight_matrix = WeightMatrix(tsp).matrix
one_tree = OneTree(weight_matrix, 0)
print(one_tree.edges)
draw(one_tree.edges, tsp)

optimization = SubgradientOptimization(weight_matrix)
