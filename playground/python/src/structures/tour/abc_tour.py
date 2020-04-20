from abc import ABC, abstractmethod
from typing import Tuple


class AbcTour(ABC):

    def __len__(self) -> int:
        """ Кол-во вершин """

    def __getitem__(self, index: int):
        """ Вершина по номеру в туре """

    def __contains__(self, edge) -> bool:
        """ Наличие ребра в туре """

    @abstractmethod
    def index(self, node) -> int:
        """ Номер вершины в туре """

    @abstractmethod
    def around(self, node) -> tuple:
        """ Предыдущая вершина и следующая текущей веришны """

    @abstractmethod
    def between(self, start, end, search) -> bool:
        """ Находится ли вершина search между вершиной start и вершиной end """

    @abstractmethod
    def successor(self, index: int):
        """ Следующий """

    @abstractmethod
    def predecessor(self, index: int):
        """ Предыдущий """

    @abstractmethod
    def generate(self, broken: set, joined: set) -> Tuple[bool, list]:
        """ Создаем новый тур, а потом проверяем его на целостность и наличие циклов
        broken: удаляемые ребра
        joined: добавляемые ребра
        """

    @abstractmethod
    def reverse(self, start, end) -> None:
        """ Переворот куска тура """
