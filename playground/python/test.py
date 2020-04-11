from typing import List, Tuple

import matplotlib.pyplot as plt

from src.lkh import LKH
from src.structures.graph import Edge
from src.structures.matrix import Matrix
from src.tsp.generator import generator


def draw(edges: List[Edge], nodes: List[Tuple[float, float]], color: str = 'r') -> None:
    for edge in edges:
        [x1, y1] = nodes[edge.dst]
        [x2, y2] = nodes[edge.src]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)

    for idx, node in enumerate(nodes):
        plt.annotate(f'{idx}:({node[0]:0.1f},{node[1]:0.1f})', node, size=9)


def candidates(alpha_matrix: Matrix, nodes: List[Tuple[float, float]], color: str = 'g') -> None:
    temp = []
    precess = 1 / alpha_matrix.dimension + 10
    print(precess)
    for idx in range(0, alpha_matrix.dimension):
        for idy in range(idx + 1, alpha_matrix.dimension):
            if alpha_matrix[idx][idy] < precess:
                temp.append((idx, idy))
    print(len(temp))

    for edge in temp:
        [x1, y1] = nodes[edge[0]]
        [x2, y2] = nodes[edge[1]]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)

    for idx, node in enumerate(nodes):
        plt.annotate(f'{idx}:({node[0]:0.1f},{node[1]:0.1f})', node, size=9)


random_tsp = [node for node in generator(50)]

lkh = LKH(random_tsp)
lkh.run()
draw(lkh.current_tour.edges(), random_tsp, 'r')
plt.show()
draw(lkh.one_tree.edges, random_tsp, 'b')
plt.show()
candidates(lkh.alpha_matrix, random_tsp)
plt.show()
