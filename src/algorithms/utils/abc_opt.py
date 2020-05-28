import logging
from abc import ABC, abstractmethod
from typing import Set, Tuple

import numpy as np

from src.algorithms.structures.collector import Collector
from src.algorithms.structures.tabu_list import TabuSet
from src.algorithms.utils.hash import generate_hash
from src.algorithms.utils.utils import get_length


class AbcOpt(ABC):
    """ Абстрактный класс, описывающий методы локального поиска
    """

    def __init__(self, length: float, tour: np.ndarray, adjacency: np.ndarray, **kwargs):
        """
        length: Текущая длина тура
        tour: Список городов
        adjacency: Матрица весов
        """
        logging.info('initialization')
        self.length, self.tour, self.matrix = length, tour, adjacency
        self.solutions: Set[int] = {generate_hash(self.tour)}
        self.size = len(tour)

        self.tabu_list = None  # проверенные ранее туры
        self.collector = None  # для сбора данных

    @abstractmethod
    def improve(self) -> float:
        """ Локальный поиск (поиск изменения + само изменение)
        return: выигрыш от локального поиска
        """

    def optimize(self) -> Tuple[float, np.ndarray]:
        """ Запуск локального поиска
        return: длина, список городов
        """
        gain, iteration, self.collector = 1, 0, Collector(['length', 'gain'], {'two_opt': self.size})
        self.collector.update({'length': self.length, 'gain': 0})
        logging.info(f'start : {self.length}')

        while gain > 0:
            gain = self.improve()
            if gain > 0:
                logging.info(f'{iteration} : {self.length}')
                iteration += 1

            h = generate_hash(self.tour)
            if h in self.solutions:
                break
            else:
                self.solutions.add(h)

            assert round(get_length(self.tour, self.matrix), 2) == round(self.length, 2), \
                f'{get_length(self.tour, self.matrix)} != {self.length}'

        return self.length, self.tour

    def meta_heuristic_optimize(self, tabu_list: TabuSet, collector: Collector) -> Tuple[float, np.ndarray]:
        """ Запуск локального поиска под управление некоторой метаэвристики
        tabu_list: проверенные ранее маршруты
        collector: структура для сбора данных о локальном поиске
        return: длина, список городов
        """
        gain, self.tabu_list, self.collector = 1, tabu_list, collector
        self.solutions = self.tabu_list.data
        self.collector.update({'length': self.length, 'gain': 0})

        while gain > 0:
            gain = self.improve()

            if gain > 1.e-10:
                self.collector.update({'length': self.length, 'gain': gain})
                if not self.tabu_list.append(self.length, self.tour):
                    break
                logging.info(self.length)

            assert round(get_length(self.tour, self.matrix), 2) == round(self.length, 2), \
                f'{get_length(self.tour, self.matrix)} != {self.length}'

        return self.length, self.tour
