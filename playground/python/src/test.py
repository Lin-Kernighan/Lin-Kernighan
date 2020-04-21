import datetime
import os
from typing import List, Tuple, Type

from src.algorithms.heuristics.abc_opt import AbcOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import Matrix
from src.tsp.generator import generator
from src.tsp.tsp_loader import TspLoader
from src.utils import draw_plots_i_y, draw_plot_x_y

Point = Tuple[float, float]


def save_test(heuristics: List[Type[AbcOpt]], names: List[str], size: int) -> None:
    tsp = [node for node in generator(size)]
    matrix = Matrix.weight_matrix(tsp)
    tour = InitialTour.greedy(matrix)

    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    name = f'{"_".join(["test", suffix])}_{size}'
    frames = []

    os.mkdir(name)
    TspLoader.python_serializer(tsp, name, name)
    heuristics = [heuristic(tour, matrix) for heuristic in heuristics]

    for heuristic in heuristics:
        heuristic.optimize()
        frames.append(heuristic.collector.as_frame())

    for frame in frames:
        frame['time'] -= frame['time'][0]

    draw_plots_i_y(frames, names, ['delta', 'gain', 'time', 'length'], name, name)
    draw_plot_x_y(frames, names, 'time', 'length', name, name)
