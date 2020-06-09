import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Tuple, Dict

import numpy as np

from lin_kernighan.algorithms.utils.abc_opt import AbcOpt
from lin_kernighan.algorithms.utils.initial_tour import greedy
from lin_kernighan.algorithms.utils.utils import get_length, mix
from lin_kernighan.utils import opts_type


def worker(opt: AbcOpt, conn: Connection, iterations: int, swap: int):
    """ Локальный поиск под управлением Поиска с запретами
    Табу обновляется раз в итерацию, в основном процессе хранится весь табу
    opt: эвристика
    conn: для передачи данных
    iterations: количество возможных перезапусков
    swap: сколько раз ломать тур за итерацию
    """
    best_length, best_tour = opt.length, opt.tour.copy()
    length, tour = opt.length, opt.tour
    try:

        while iterations > 0:
            _length, _tour = opt.optimize()
            if best_length > _length:
                best_length, best_tour = _length, _tour.copy()
                assert round(get_length(best_tour, opt.matrix), 2) == round(best_length, 2), \
                    f'{get_length(best_tour, opt.matrix)} != {best_length}'

            conn.send(opt.solutions)
            solutions = conn.recv()

            mix(tour, swap)
            length = get_length(tour, opt.matrix)
            opt.length, opt.tour, opt.solutions = length, tour, solutions
            iterations -= 1

    except Exception as exc:
        print(f'Exception: {exc}')

    conn.send(set())
    conn.send(best_length)
    conn.send(best_tour)


class TabuProcSearch:
    """ Базовая метаэвристика: многопроцессорный Поиск с запретами
    opt: название эвристики поиска [two_opt, three_opt, lk, lkh]
    matrix: матрица весов
    proc: количество процессов
    **kwargs: дополнительные параметры для локального поиска
    """

    def __init__(self, opt: str, matrix: np.ndarray, **kwargs):
        length, tour = greedy(matrix)
        self.matrix = matrix
        self.opt = opts_type[opt](length, tour, matrix, **kwargs)
        self.length, self.tour = self.opt.length, self.opt.tour
        self.proc = kwargs.get('proc', 4)

    def optimize(self, iterations=10, swap=2) -> Tuple[float, np.ndarray]:
        """ Запуск метаэвристики табу поиска на нескольких процессах
        Алгоритм запоминает все локальные минимумы: all_solutions
        iteration: количество возможных перезапусков
        swap: сколько раз ломать тур за итерацию. Если тур не улучшится, на следующей итерации ломается он же
        return: лучшая длина тура, лучший тур
        """
        processes: Dict[mp.Process, Connection] = {}
        pid_process: Dict[int, int] = {}
        logging.info(f'start: {self.length}')

        for idx in range(self.proc):
            m, w = mp.Pipe()
            p = mp.Process(target=worker, args=(self.opt, w, iterations, swap))
            p.start()
            processes[p], pid_process[p.pid] = m, idx

        all_solutions = set()
        while True:
            for proc, conn in processes.items():
                if conn.poll():
                    solutions = conn.recv()
                    if len(solutions) == 0:
                        length, tour = conn.recv(), conn.recv()
                        if length < self.length:
                            self.length, self.tour = length, tour
                        proc.join()
                        del processes[proc]
                        logging.info(f'Done: {pid_process[proc.pid]} - {length}')
                        break
                    all_solutions |= solutions
                    conn.send(all_solutions)
                    logging.info(f'Update: {pid_process[proc.pid]}')

            if len(processes) == 0:
                break

        logging.info(f'tabu search done, best length: {self.length}')
        return self.length, self.tour
