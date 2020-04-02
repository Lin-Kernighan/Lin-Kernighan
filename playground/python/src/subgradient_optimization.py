from sys import maxsize
from typing import List

import numpy as np

from src.minimum_spanning_tree import MinimumSpanningTree
from src.graph import Edge


class SubgradientOptimization:
    weight_matrix: List[List[float]]
    pi_max: List[float]
    length: int

    def __init__(self, weight_matrix: List[List[float]], max_iterations=1000):
        self.weight_matrix = weight_matrix
        self.length = len(weight_matrix)

        pi, w_max, w = np.zeros(self.length), -maxsize, -maxsize  # инициализируем итерацию, штрафы и текущий максимум
        self.pi_max = pi[:]
        v = np.zeros(self.length)

        t = 0.0001
        period = next_period = self.length // 2
        is_first_period = True
        is_increasing = True
        last_improve = 0
        best_solution = 0

        for k in range(1, max_iterations):
            self.__make_move(pi)
            mst = MinimumSpanningTree(self.weight_matrix)
            ll = mst.total_price  # получаем длину нового деревого
            w_prev, w = w, ll - 2 * pi.sum()  # считаем полученную длину

            if w > w_max + 1e-6:  # максимальная пока что длина
                w_max, self.pi_max = w, pi.copy()
                last_improve = k

            v_prev, v = v, self.__get_degrees(mst.edges)  # получаем субградиенты

            # -------------------- обновляем pi -----------------------------------------------------
            pi = pi + t * (0.7 * v + 0.3 * v_prev)
            # ic(k, w_max, w, t, period, ll, 2*pi.sum(), last_improve)
            if best_solution < self.__check(v):
                best_solution = self.__check(v)
                print(f'new best:{best_solution}, iter:{k}')
            # print(f'{k}:{v}:{self.__check(v)}')
            # ic(k, w_max, w, pi, t, period, ll, v)

            # --------------------- магия с шагом оптимизации ---------------------------------------
            period -= 1
            if is_first_period and is_increasing:  # случай когда мы вначале увеличиваем шаг
                if w <= w_prev:
                    is_increasing = False
                    t /= 2
                else:
                    t *= 2

            if k - last_improve >= 10:  # вставка чтобы избежать стагнации
                t /= 1.07

            if period == 0:  # случай когда период закончился
                is_first_period = False
                next_period = next_period // 2  # уменьшаем в два раза длину периода
                t /= 2  # и уменьшаем размер шага
                if k - last_improve <= 2:  # пункт с удвоением, если все идет хорошо
                    next_period = next_period * 2
                    t *= 2
                period = next_period

            if period == 0 or t < 1e-10 or np.absolute(v).sum() == 0:  # условие выхода
                print(f't:{t:0.10f}, period:{period}, abs.sum:{np.absolute(v).sum()}, sol:{self.__check(v)}, iter:{k}')
                break

        print(last_improve)

    def __make_move(self, pi: np.ndarray) -> None:
        for i, k in enumerate(pi):
            for index in range(self.length):
                self.weight_matrix[i][index] += k
                self.weight_matrix[index][i] += k

    def __get_degrees(self, edges: List[Edge]) -> np.ndarray:
        v = np.asarray([-2] * self.length)
        for edge in edges:
            v[edge.dst] += 1
            v[edge.src] += 1
        return v

    @staticmethod
    def __check(v: np.ndarray) -> int:
        i = 0
        for num in v:
            if num == 0:
                i += 1
        return i
