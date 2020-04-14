from typing import List, Tuple

import matplotlib.pyplot as plt

from src.structures.graph import Edge
from src.structures.matrix import Matrix


def draw(edges: List[Edge], nodes: List[Tuple[float, float]], color: str) -> None:
    for edge in edges:
        [x1, y1] = nodes[edge.dst]
        [x2, y2] = nodes[edge.src]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)

    for idx, node in enumerate(nodes):
        plt.annotate(f'{idx}:({node[0]:0.1f},{node[1]:0.1f})', node, size=9)


def draw_tour(tour: List[int], nodes: List[Tuple[float, float]], color: str) -> None:
    first, second = tour[-1], tour[0]
    [x1, y1] = nodes[first]
    [x2, y2] = nodes[second]
    plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)
    for idx in range(len(tour) - 1):
        [x1, y1] = nodes[tour[idx]]
        [x2, y2] = nodes[tour[idx + 1]]
        plt.plot([x1, x2], [y1, y2], linewidth=1, color=color)
    for idx, node in enumerate(nodes):
        plt.annotate(f'{idx}:({node[0]:0.1f},{node[1]:0.1f})', node, size=9)


def candidates(alpha_matrix: Matrix, nodes: List[Tuple[float, float]], color: str, one_tree: float) -> None:
    temp = []
    precess = 1 / alpha_matrix.dimension * one_tree
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
