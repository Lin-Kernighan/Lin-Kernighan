import logging
from typing import Tuple

import numpy as np

from src.algorithms.utils.abc_search import AbcSearch
from src.algorithms.utils.utils import get_length, mix
from src.utils import opts_type


class TabuSearch(AbcSearch):
    """ Базовая метаэвристика: Поиск с запретами
    opt: название эвристики поиска [two_opt, three_opt, lk_opt, lkh_opt]
    matrix: матрица весов

    **kwargs: дополнительные параметры для локального поиска
    """

    def __init__(self, opt: str, matrix: np.ndarray, **kwargs):
        super().__init__(matrix, **kwargs)
        self.opt = opts_type[opt](self.length, self.tour, self.matrix, **kwargs)
        logging.info('initialization tabu search done')

    def optimize(self, iterations=10, swap=2, **kwargs) -> Tuple[float, np.ndarray]:
        """ Запуск метаэвристики табу поиска
        Алгоритм запоминает все локальные минимумы, и, если попадает в одно из  них, перезапускает поиск
        iterations: количество возможных перезапусков
        swap: сколько раз ломать тур за итерацию. Если тур не улучшится, на следующей итерации ломается он же
        return: лучшая длина тура, лучший тур
        """
        if self.collector is not None:
            self.collector.update({'length': self.length, 'gain': 0})

        while iterations > 0:
            self.opt.meta_heuristic_optimize(self.data, self.collector)
            _length, _tour = self.best_tour()

            if self.length > _length:
                self.length, self.tour = _length, _tour  # если найден другой хороший оптимум, обновляем текущий
                assert round(get_length(self.tour, self.matrix), 2) == round(self.length, 2), \
                    f'{get_length(self.tour, self.matrix)} != {self.length}'

            logging.info(f'{iterations} : {_length} : {self.length}')
            mix(self.tour, swap)  # а вот ломается текущий сохраненный
            self.length = get_length(self.tour, self.matrix)
            self.opt.length, self.opt.tour = self.length, self.tour.copy()  # улучшается только копия
            iterations -= 1

        self.length, self.tour = self.best_tour()
        logging.info(f'tabu search done, best length: {self.length}')
        return self.length, self.tour
