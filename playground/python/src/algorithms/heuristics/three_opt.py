from typing import List, Tuple

from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import Matrix
from src.utils import right_rotate, get_length

Point = Tuple[float, float]


class ThreeOpt:

    @staticmethod
    def run(points: List[Point]) -> List[int]:
        """ Полный запуск на точках """
        matrix = Matrix.weight_matrix(points)
        init_tour = InitialTour.greedy(matrix)
        return ThreeOpt.optimize(init_tour, matrix)

    @staticmethod
    def optimize(tour: List[int], matrix: Matrix) -> List[int]:
        """ Запуск на готовом туре и матрице смежностей """
        best_gain = 1
        iteration = 0
        length = get_length(matrix, tour)
        print(f'start : {length}')
        while best_gain > 0:
            best_gain, tour = ThreeOpt.__three_opt(matrix, tour)
            if best_gain == 0:
                tour = right_rotate(tour, len(tour) // 3)  # костыль, велосипедов пока не завезли
                best_gain, tour = ThreeOpt.__three_opt(matrix, tour)
            length -= best_gain
            print(f'{iteration} : {length}')  # оставлю, чтобы видеть, что алгоритм не помер еще
            iteration += 1
        return tour

    @staticmethod
    def __three_opt(matrix: Matrix, tour: List[int]) -> Tuple[float, List[int]]:
        """ 3-opt """
        best_exchange, best_gain, best_nodes = ThreeOpt.__improve(matrix, tour)
        if best_gain > 0:
            tour = ThreeOpt.__exchange(tour, best_exchange, best_nodes)
        return best_gain, tour

    @staticmethod
    def __improve(matrix: Matrix, tour: List[int]) -> Tuple[int, float, tuple]:
        """ 3-opt пробег по вершинам """
        best_exchange, best_gain, best_nodes = 0, 0, None
        size = matrix.dimension

        for x in range(size - 5):
            for y in range(x + 2, size - 3):
                for z in range(y + 2, size - 1):
                    exchange, gain = ThreeOpt.__search(matrix, tour, x, y, z)
                    if gain > best_gain:
                        best_gain, best_exchange, best_nodes = gain, exchange, (x, y, z)

        return best_exchange, best_gain, best_nodes

    @staticmethod
    def __search(matrix: Matrix, tour: List[int], x: int, y: int, z: int) -> Tuple[int, float]:
        """ Поиск лучшего среди переборов """
        a, b, c, d, e, f = tour[x], tour[x + 1], tour[y], tour[y + 1], tour[z], tour[z + 1]
        base = current_min = matrix[a][b] + matrix[c][d] + matrix[e][f]
        gain = 0
        exchange = -1

        if current_min > (current := matrix[a][e] + matrix[c][d] + matrix[b][f]):  # 2-opt (a, e) [d, c] (b, f)
            gain, exchange, current_min = base - current, 0, current
        if current_min > (current := matrix[a][b] + matrix[c][e] + matrix[d][f]):  # 2-opt [a, b] (c, e) (d, f)
            gain, exchange, current_min = base - current, 1, current
        if current_min > (current := matrix[a][c] + matrix[b][d] + matrix[e][f]):  # 2-opt (a, c) (b, d) [e, f]
            gain, exchange, current_min = base - current, 2, current
        if current_min > (current := matrix[a][d] + matrix[e][c] + matrix[b][f]):  # 3-opt (a, d) (e, c) (b, f)
            gain, exchange, current_min = base - current, 3, current
        if current_min > (current := matrix[a][d] + matrix[e][b] + matrix[c][f]):  # 3-opt (a, d) (e, b) (c, f)
            gain, exchange, current_min = base - current, 4, current
        if current_min > (current := matrix[a][e] + matrix[d][b] + matrix[c][f]):  # 3-opt (a, e) (d, b) (c, f)
            gain, exchange, current_min = base - current, 5, current
        if current_min > (current := matrix[a][c] + matrix[b][e] + matrix[d][f]):  # 3-opt (a, c) (b, e) (d, f)
            gain, exchange, current_min = base - current, 6, current

        return exchange, gain

    @staticmethod
    def __exchange(tour: List[int], best_exchange: int, nodes: tuple) -> List[int]:
        """ Конечная замена """
        x, y, z = nodes
        a, b, c, d, e, f = x, x + 1, y, y + 1, z, z + 1
        sol = []
        if best_exchange == 0:
            sol = tour[:a + 1] + tour[e:d - 1:-1] + tour[c:b - 1:-1] + tour[f:]
        elif best_exchange == 1:
            sol = tour[:a + 1] + tour[b:c + 1] + tour[e:d - 1:-1] + tour[f:]
        elif best_exchange == 2:
            sol = tour[:a + 1] + tour[c:b - 1:-1] + tour[d:e + 1] + tour[f:]
        elif best_exchange == 3:
            sol = tour[:a + 1] + tour[d:e + 1] + tour[c:b - 1:-1] + tour[f:]
        elif best_exchange == 4:
            sol = tour[:a + 1] + tour[d:e + 1] + tour[b:c + 1] + tour[f:]
        elif best_exchange == 5:
            sol = tour[:a + 1] + tour[e:d - 1:-1] + tour[b:c + 1] + tour[f:]
        elif best_exchange == 6:
            sol = tour[:a + 1] + tour[c:b - 1:-1] + tour[e:d - 1:-1] + tour[f:]
        return sol
