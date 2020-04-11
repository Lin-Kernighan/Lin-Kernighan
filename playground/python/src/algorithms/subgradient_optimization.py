from __future__ import annotations

from sys import maxsize
from typing import List

import numpy as np

from src.structures.graph import Edge
from src.structures.matrix import Matrix
from src.structures.one_tree import OneTree


class SubgradientOptimization:
    pi_max: np.ndarray
    pi_sum: np.ndarray
    w_max: float

    @staticmethod
    def run(weight_matrix: Matrix, max_iterations=500) -> SubgradientOptimization:
        opt = SubgradientOptimization()
        length = len(weight_matrix)

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
            SubgradientOptimization.make_move(pi, weight_matrix)
            one_tree = OneTree.build(weight_matrix)  # с нулевой вершиной
            ll = one_tree.total_price  # получаем длину нового деревого
            w_prev, w = w, ll - 2 * pi.sum()  # считаем полученную длину

            if w > opt.w_max + 1e-6:  # максимальная пока что длина
                opt.w_max, opt.pi_max, opt.pi_sum = w, pi.copy(), pi_sum.copy()
                last_improve = k

            v_prev, v = v, opt.__get_degrees(one_tree.edges, length)  # получаем субградиенты

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
        SubgradientOptimization.get_back(pi_sum - pi, weight_matrix)
        return opt

    @staticmethod
    def make_move(pi: np.ndarray, weight_matrix: Matrix) -> None:
        """ vertex pi[i] added to all elements of i-row and i-column of weight matrix
        """
        length = len(weight_matrix)
        for i, k in enumerate(pi):
            for index in range(length):
                weight_matrix[i][index] += k
                weight_matrix[index][i] += k

    @staticmethod
    def get_back(pi: np.ndarray, weight_matrix: Matrix) -> None:
        """ get matrix before move
        """
        length = len(weight_matrix)
        for i, k in enumerate(pi):
            for index in range(length):
                weight_matrix[i][index] -= k
                weight_matrix[index][i] -= k

    @staticmethod
    def __get_degrees(edges: List[Edge], length: int) -> np.ndarray:
        """ v^k = d^k - 2,
        where d is vector having as its elements the
        degrees of the nodes in the current minimum 1-tree
        """
        v = np.asarray([-2] * length)
        for edge in edges:
            v[edge.dst] += 1
            v[edge.src] += 1
        return v

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'pi_max:{self.pi_max}\nw_max:{self.w_max}'
