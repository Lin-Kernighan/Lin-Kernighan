from typing import List

import matplotlib.pyplot as plt

from src.lkh import LKH
from src.structures.graph import Edge
from src.tsp.oliver30 import tsp


def draw(edges: List[Edge], nodes: List[List[float]]):
    for edge in edges:
        [x1, y1] = nodes[edge.dst]
        [x2, y2] = nodes[edge.src]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color='r')

    for idx, node in enumerate(nodes):
        plt.annotate(f'{idx}:({node[0]},{node[1]})', node, size=9)
    plt.show()


lkh = LKH(tsp)
lkh.run()
