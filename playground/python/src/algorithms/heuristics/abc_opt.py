import logging
from abc import ABC, abstractmethod
from typing import Set

import numpy as np

from src.structures.collector import Collector
from src.structures.tabu_list import TabuSet
from src.utils import get_length, get_hash


class AbcOpt(ABC):
    """ Абстрактный класс, описывающий методы локального поиска
    """

    def __init__(self, length: float, tour: np.ndarray, adjacency: np.ndarray, **kwargs):
        """
        length: Текущая длина тура
        tour: Список городов
        adjacency: Матрица весов
        """
        self.length, self.tour, self.matrix = length, tour, adjacency
        self.solutions: Set[int] = {get_hash(self.tour)}
        self.size = len(tour)
        self.tabu_list = None  # проверенные ранее туры
        self.collector = None  # для сбора данных

    @abstractmethod
    def improve(self) -> float:
        """ Локальный поиск (поиск изменения + само изменение)
        return: выигрыш от локального поиска
        """

    def optimize(self) -> np.ndarray:
        """ Запуск локального поиска
        return: новый маршрут
        """
        gain, iteration, self.collector = 1, 0, Collector(['length', 'gain'], {'two_opt': self.size})
        self.collector.update({'length': self.length, 'gain': 0})
        logging.info(f'start : {self.length}')

        while gain > 0:
            gain = self.improve()
            if gain > 0:
                logging.info(f'{iteration} : {self.length}')
                iteration += 1

            # h = get_hash(self.tour)
            # if h in self.solutions:
            #     break
            # else:
            #     self.solutions.add(h)

            assert round(get_length(self.matrix, self.tour), 6) == round(self.length, 6), \
                f'{get_length(self.matrix, self.tour)} != {self.length}'

        return self.tour

    def tabu_optimize(self, tabu_list: TabuSet, collector: Collector) -> np.ndarray:
        """ Запуск локального поиска под управление tabu search
        tabu_list: проверенные ранее маршруты
        collector: структура для сбора данных о локальном поиске
        return: новый маршрут
        """
        self.tabu_list, best_change, self.collector = tabu_list, -1, collector
        self.collector.update({'length': self.length, 'gain': 0})
        logging.info(f'Start: {self.length}')

        while best_change > 0:
            gain = self.improve()
            if gain > 0.0:
                if self.tabu_list.contains(self.tour):
                    break
                self.length -= gain
                tabu_list.append(self.tour, self.length)
                self.collector.update({'length': self.length, 'gain': -best_change})

        logging.info(f'End: {self.length}')
        return self.tour
