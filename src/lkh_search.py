import logging
from typing import Tuple

import numpy as np

from src.algorithms.lkh_opt import LKHOpt
from src.algorithms.two_opt import TwoOpt
from src.algorithms.utils.abc_search import AbcSearch
from src.algorithms.utils.initial_tour import helsgaun, fast_helsgaun, greedy, two_opt
from src.algorithms.utils.utils import get_length, get_set

_initialization = dict(helsgaun=helsgaun, fast_helsgaun=fast_helsgaun, greedy=greedy, two_opt=two_opt)


class LKHSearch(AbcSearch):
    """ Базовая метаэвристика: Multi trial LKH
    matrix: матрица весов

    init: генерация нового тура [helsgaun, fast_helsgaun, greedy, two_opt]
    dlb: don't look bits [boolean]
    bridge: make double bridge [boolean]
    excess: parameter for cut bad candidates [float]
    mul: excess factor [float]
    two_opt: use two_opt by initial tour [boolean]
    non_seq: use non sequential move [boolean]
    k: number of k for k-opt; how many sequential can make algorithm [int]
    subgradient: use or not subgradient optimization [boolean]
    """

    def __init__(self, matrix: np.ndarray, **kwargs):
        super().__init__(matrix, **kwargs)

        if kwargs.get('two_opt', False):
            self.length, self.tour = TwoOpt.just_improve(self.length, self.tour, self.matrix)

        self.opt = LKHOpt(self.length, self.tour, self.matrix, **kwargs)
        self.initial = kwargs.get('init', 'fast_helsgaun')

        logging.info('initialization multi trial lkh done')

    def optimize(self, iterations=10, **kwargs) -> Tuple[float, np.ndarray]:
        """ Запуск метаэвристики Multi trial LKH
        iterations: количество возможных перезапусков
        return: лучшая длина тура, лучший тур
        """
        if self.collector is not None:
            self.collector.update({'length': self.length, 'gain': 0})

        while iterations > 0:
            self.opt.meta_heuristic_optimize(self.data, self.collector)
            _length, _tour = self.best_tour()

            if self.length > _length:
                self.length, self.tour = _length, _tour  # если найден другой хороший оптимум, обновляем текущий
                self.opt.best_solution = get_set(self.tour)
                assert round(get_length(self.tour, self.matrix), 2) == round(self.length, 2), \
                    f'{get_length(self.tour, self.matrix)} != {self.length}'

            logging.info(f'{iterations} : {_length} : {self.length}')

            if self.initial == 'helsgaun' or self.initial == 'fast_helsgaun':
                self.opt.length, self.opt.tour = _initialization[self.initial](
                    self.opt.alpha,
                    self.matrix,
                    self.opt.best_solution,
                    self.opt.candidates,
                    self.opt.excess)
            else:
                self.opt.length, self.opt.tour = _initialization[self.initial](self.matrix)

            assert round(get_length(self.opt.tour, self.matrix), 2) == round(self.opt.length, 2), \
                f'{get_length(self.opt.tour, self.matrix)} != {self.opt.length}'

            iterations -= 1

        self.length, self.tour = self.best_tour()
        logging.info(f'multi trial lkh done, best length: {self.length}')
        return self.length, self.tour
