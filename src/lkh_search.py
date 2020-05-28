import logging
from typing import Tuple

import numpy as np

from src.algorithms.lkh_opt import LKHOpt
from src.algorithms.utils.abc_search import AbcSearch
from src.algorithms.utils.initial_tour import helsgaun, fast_helsgaun
from src.algorithms.utils.utils import get_length, get_set

_initialization = dict(helsgaun=helsgaun, fast_helsgaun=fast_helsgaun)


class LKHSearch(AbcSearch):
    """ Базовая метаэвристика: Multi trial LKH
    matrix: матрица весов

    init: генерация нового тура [helsgaun, fast_helsgaun]

    dlb: don't look bits [boolean]
    bridge: make double bridge [tuple] ([not use: 0, all cities: 1, only neighbours: 2], fast scheme)
    excess: parameter for cut bad candidates [float]
    mul: excess factor
    k: number of k for k-opt; how many sequential can make algorithm [int]
    subgradient: use or not subgradient optimization
    """

    def __init__(self, matrix: np.ndarray, **kwargs):
        super().__init__(matrix, **kwargs)
        self.opt = LKHOpt(self.length, self.tour, self.matrix, **kwargs)
        self.initial = kwargs.get('init', 'fast_helsgaun')

        logging.info('initialization multi trial lkh done')

    def optimize(self, iteration=10, **kwargs) -> Tuple[float, np.ndarray]:
        """ Запуск метаэвристики Multi trial LKH
        iteration: количество возможных перезапусков
        return: лучшая длина тура, лучший тур
        """
        self.collector.update({'length': self.length, 'gain': 0})

        while iteration > 0:
            self.opt.meta_heuristic_optimize(self.data, self.collector)
            _length, _tour = self.best_tour()

            if self.length > _length:
                self.length, self.tour = _length, _tour  # если найден другой хороший оптимум, обновляем текущий
                self.opt.best_solution = get_set(self.tour)
                assert round(get_length(self.tour, self.matrix), 2) == round(self.length, 2), \
                    f'{get_length(self.tour, self.matrix)} != {self.length}'

            logging.info(f'{iteration} : {_length} : {self.length}')
            self.opt.length, self.opt.tour = _initialization[self.initial](
                self.opt.alpha,
                self.matrix,
                self.opt.best_solution,
                self.opt.candidates,
                self.opt.excess
            )
            assert round(get_length(self.opt.tour, self.matrix), 2) == round(self.opt.length, 2), \
                f'{get_length(self.opt.tour, self.matrix)} != {self.opt.length}'
            iteration -= 1

        self.length, self.tour = self.best_tour()
        logging.info(f'multi trial lkh done, best length: {self.length}')
        return self.length, self.tour
