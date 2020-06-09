from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from lin_kernighan.algorithms.structures.collector import Collector
from lin_kernighan.algorithms.structures.tabu_list import TabuSet
from lin_kernighan.algorithms.utils.initial_tour import greedy


class AbcSearch(ABC):
    """ Абстрактный класс, описывающий методы метаэвристик поиска
    """

    def __init__(self, matrix: np.ndarray, **kwargs):
        (self.length, self.tour), self.matrix = greedy(matrix), matrix

        collect = kwargs.get('collect', False)
        self.collector = None if not collect else Collector(['length', 'gain'], {'lkh_search': len(self.tour)})

        self.data = TabuSet()
        self.data.append(self.length, self.tour)

    @abstractmethod
    def optimize(self, iteration: int, **kwargs) -> Tuple[float, np.ndarray]:
        """ Запуск метаэвристики """

    def best_tour(self) -> Tuple[float, np.ndarray]:
        """ Возвращает лучший найденный тур
        return: лучшая длина тура, лучший тур
        """
        return self.data.best_tour()
