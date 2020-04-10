from typing import List, Tuple

from src.structures.one_tree import OneTree


class AlphaNearness:
    weight_matrix: List[List[float]]
    optimal_price: float

    def __init__(self, weight_matrix: List[List[float]], node: int = 0) -> None:
        self.weight_matrix = weight_matrix
        self.optimal_price = OneTree(weight_matrix, node).total_price

    def alpha_nearness(self, with_edge: Tuple[int, int], node: int = 0) -> float:
        return OneTree(self.weight_matrix, node, with_edge).total_price - self.optimal_price
