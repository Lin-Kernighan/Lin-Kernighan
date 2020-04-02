from typing import List

import matplotlib.pyplot as plt

from src.utils import Edge


class Graph:
    edges: List[Edge]
    nodes: List[List[float]]

    def __init__(self, points: List[List[float]]) -> None:
        self.nodes = points
        self.edges = []

    def draw(self):
        for edge in self.edges:
            [x1, y1] = self.nodes[edge.dst]
            [x2, y2] = self.nodes[edge.src]
            plt.plot([x1, x2], [y1, y2], linewidth=1, color='r')

        for idx, node in enumerate(self.nodes):
            plt.annotate(f'{idx}:({node[0]},{node[1]})', node, size=9)
        plt.show()
