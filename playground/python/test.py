from typing import List, Tuple

import matplotlib.pyplot as plt

from src.lkh import LKH
from src.structures.graph import Edge
from src.tsp.generator import generator


def draw(edges: List[Edge], nodes: List[Tuple[float, float]]) -> None:
    for edge in edges:
        [x1, y1] = nodes[edge.dst]
        [x2, y2] = nodes[edge.src]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color='r')

    for idx, node in enumerate(nodes):
        plt.annotate(f'{idx}:({node[0]:0.1f},{node[1]:0.1f})', node, size=9)
    plt.show()


random_tsp = [node for node in generator(50)]
print(random_tsp)

lkh = LKH(random_tsp)
lkh.run()
draw(lkh.current_tour.edges(), random_tsp)
