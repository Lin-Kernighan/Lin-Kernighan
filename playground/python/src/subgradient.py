# import numpy as np
#
# from spanning_trees.minimal_spanning_tree import get_tree_len, get_degs
#
#
# def step_gen(n, w_max, w_pi):
#     period = n / 2
#     step_size = 1
#     for i in range(period):
#         yield step_size
#         if w_max <= w_pi:
#             pass
#
#     period //= 2
#     step_size /= 2
#
#
# def get_step_size(n, k, w_prev, w_pi):
#     return 1. / (k ** 0.5 + 200)
#
#
# # @jit(parallel=True)
# def subgrad_opt(x, max_iters=100):
#     """ Subgradient optimization """
#     n = x.shape[0]
#     pi, w_max, w = np.zeros(n), -np.inf, -np.inf  # инициализируем итерацию, штрафы и текущий максимум
#     pi_max = pi[:]
#     v = np.zeros(n)
#
#     t = 0.0001
#     period = next_period = n // 2
#     is_first_period = True
#     is_increasing = True
#     last_improve = 0
#
#     for k in range(1, max_iters):
#         ll, tree = get_tree_len(x, pi)  # получаем новое дерево и его длину
#         w_prev, w = w, ll - 2 * pi.sum()  # считаем полученную длину
#
#         if w > w_max + 1e-6:  # максимальная пока что длина
#             w_max, pi_max = w, pi.copy()
#             last_improve = k
#
#         v_prev, v = v, get_degs(tree)  # получаем субградиенты
#
#         # -------------------- обновляем pi -----------------------------------------------------
#         pi = pi + t * (0.7 * v + 0.3 * v_prev)
#         # ic(k, w_max, w, t, period, ll, 2*pi.sum(), last_improve)
#         print(ll)
#         # ic(k, w_max, w, pi, t, period, ll, v)
#
#         # --------------------- магия с шагом оптимизации ---------------------------------------
#         period -= 1
#
#         if is_first_period and is_increasing:  # случай когда мы вначале увеличиваем шаг
#             if w <= w_prev:
#                 is_increasing = False
#                 t /= 2
#             else:
#                 t *= 2
#
#         if k - last_improve >= 10:  # вставка чтобы избежать стогнации
#             t /= 1.07
#
#         if period == 0:  # случай когда период закончился
#             is_first_period = False
#             next_period = next_period // 2  # уменьшаем в два раза длину периода
#             t /= 2  # и уменьшаем размер шага
#
#             if k - last_improve <= 2:  # пункт с удвоением, если все идет хорошо
#                 next_period = next_period * 2
#                 t *= 2
#
#             period = next_period
#
#         if period == 0 or t < 1e-10 or np.absolute(v).sum() == 0:  # условие выхода
#             return pi_max
#
#     return pi_max
