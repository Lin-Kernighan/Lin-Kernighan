from __future__ import annotations

from sys import maxsize
from typing import Tuple

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

Edge = Tuple[int, int]


class SubgradientOptimization:
    pi_max: np.ndarray
    pi_sum: np.ndarray
    w_max: float

    @staticmethod
    def run(adjacency_matrix: np.ndarray, max_iterations=100) -> SubgradientOptimization:
        opt = SubgradientOptimization()
        length = adjacency_matrix.shape[0]

        pi = np.zeros(length)  # итерацию
        pi_sum = np.zeros(length)
        v = np.zeros(length)

        opt.w_max, w = -maxsize, -maxsize  # инициализируем текущий максимум и штрафы
        opt.pi_max = pi[:]

        t = 0.0001
        period = next_period = length // 2
        is_first_period = True
        is_increasing = True
        last_improve = 0

        for k in range(1, max_iterations):
            SubgradientOptimization.make_move(pi, adjacency_matrix)
            ll, first, second = SubgradientOptimization.__one_tree(adjacency_matrix)  # получаем длину нового деревого
            w_prev, w = w, ll - 2 * pi.sum()  # считаем полученную длину

            if w > opt.w_max + 1e-6:  # максимальная пока что длина
                opt.w_max, opt.pi_max, opt.pi_sum = w, pi.copy(), pi_sum.copy()
                last_improve = k

            v_prev, v = v, opt.__get_degrees(first, second, length)  # получаем субградиенты

            # -------------------- обновляем pi -----------------------------------------------------
            pi = pi + t * (0.7 * v + 0.3 * v_prev)
            pi_sum += pi
            # ic(k, w_max, w, t, period, ll, 2*pi.sum(), last_improve)
            # ic(k, w_max, w, pi, t, period, ll, v)

            # --------------------- магия с шагом оптимизации ---------------------------------------
            period -= 1
            if is_first_period and is_increasing:  # случай когда мы вначале увеличиваем шаг
                if w <= w_prev:
                    is_increasing = False
                    t /= 2
                else:
                    t *= 2

            # if k - last_improve >= 10:  # вставка чтобы избежать стагнации
            #     t /= 1.07

            if period == 0:  # случай когда период закончился
                is_first_period = False
                next_period = next_period // 2  # уменьшаем в два раза длину периода
                t /= 2  # и уменьшаем размер шага
                if k - last_improve <= 2:  # пункт с удвоением, если все идет хорошо
                    next_period = next_period * 2
                    t *= 2
                period = next_period

            if period == 0 or t < 1e-10 or np.absolute(v).sum() == 0:  # условие выхода
                break
        SubgradientOptimization.get_back(pi_sum - pi, adjacency_matrix)
        return opt

    @staticmethod
    @njit
    def make_move(pi: np.ndarray, adjacency_matrix: np.ndarray) -> None:
        """ vertex pi[i] added to all elements of i-row and i-column of adjacency matrix
        """
        for i, k in enumerate(pi):
            for index in range(adjacency_matrix.shape[0]):
                adjacency_matrix[i][index] += k
                adjacency_matrix[index][i] += k

    @staticmethod
    @njit
    def get_back(pi: np.ndarray, adjacency_matrix: np.ndarray) -> None:
        """ get matrix before move
        """
        for i, k in enumerate(pi):
            for index in range(adjacency_matrix.shape[0]):
                adjacency_matrix[i][index] -= k
                adjacency_matrix[index][i] -= k

    @staticmethod
    @njit
    def __get_degrees(first: np.ndarray, second: np.ndarray, length: int) -> np.ndarray:
        """ v^k = d^k - 2,
        where d is vector having as its elements the
        degrees of the nodes in the current minimum 1-tree
        """
        v = np.asarray([-2] * length)
        for idx in range(length):
            v[first[idx]] += 1
            v[second[idx]] += 1
        return v

    @staticmethod
    def __one_tree(adjacency_matrix: np.ndarray, node: int = 0) -> Tuple[float, np.ndarray, np.ndarray]:
        # noinspection PyTypeChecker
        mst: csr_matrix = minimum_spanning_tree(adjacency_matrix)
        coo = mst.tocoo()
        first, second, temp = coo.col, coo.row, coo.data.sum()

        indexes = [second[idx] for idx in np.where(first == node)[0]] + \
                  [first[idx] for idx in np.where(second == node)[0]]

        that, minimum = -1, float('inf')
        for idx, value in enumerate(adjacency_matrix[node]):
            if node == idx or not value > 0:
                continue
            if value < minimum and idx not in indexes:
                that, minimum = idx, value

        return temp + minimum, np.append(first, node), np.append(second, that)
