from sys import maxsize
from typing import List

import numpy as np

from src.structures.graph import Edge
from src.structures.matrix import Matrix
from src.structures.one_tree import OneTree


class SubgradientOptimization:
    pi_max: List[float]
    w_max: float
    length: int

    def __init__(self, weight_matrix: Matrix, max_iterations=1000):
        length = len(weight_matrix)
        pi = np.zeros(length)  # итерацию
        v = np.zeros(length)

        self.w_max, w = -maxsize, -maxsize  # инициализируем текущий максимум и штрафы
        self.pi_max = pi[:]

        t = 0.0001
        period = next_period = length // 2
        is_first_period = True
        is_increasing = True
        last_improve = 0

        for k in range(1, max_iterations):
            self.__make_move(pi, weight_matrix)
            one_tree = OneTree(weight_matrix, 0)  # с нулевой вершиной
            ll = one_tree.total_price  # получаем длину нового деревого
            w_prev, w = w, ll - 2 * pi.sum()  # считаем полученную длину

            if w > self.w_max + 1e-6:  # максимальная пока что длина
                self.w_max, self.pi_max = w, pi.copy()
                last_improve = k

            v_prev, v = v, self.__get_degrees(one_tree.edges, length)  # получаем субградиенты

            # -------------------- обновляем pi -----------------------------------------------------
            pi = pi + t * (0.7 * v + 0.3 * v_prev)
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

    @staticmethod
    def __make_move(pi: np.ndarray, weight_matrix: Matrix) -> None:
        """ vertex pi[i] added to all elements of i-row and i-column of weight matrix
        """
        length = len(weight_matrix)
        for i, k in enumerate(pi):
            for index in range(length):
                weight_matrix[i][index] += k
                weight_matrix[index][i] += k

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
