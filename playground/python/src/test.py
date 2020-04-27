import datetime
import os
from typing import List, Tuple, Type

from src.algorithms.heuristics.abc_opt import AbcOpt
from src.algorithms.initial_tour import InitialTour
from src.structures.matrix import adjacency_matrix
from src.tsp.generator import generator
from src.tsp.tsp_loader import TspLoader
from src.utils import draw_plots_i_y, draw_plot_x_y

Point = Tuple[float, float]


def save_test(heuristics: List[Type[AbcOpt]], names: List[str], size: int) -> None:
    tsp = generator(size)
    matrix = adjacency_matrix(tsp)
    length, tour = InitialTour.greedy(matrix)

    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = f'{"_".join(["test", suffix])}_{size}'
    frames = []

    os.mkdir(filename)
    TspLoader.python_serializer(tsp, filename, filename)
    heuristics = [heuristic(tour, matrix) for heuristic in heuristics]

    for idx in range(len(heuristics)):
        heuristics[idx].optimize()
        frames.append(heuristics[idx].collector.as_frame())
        heuristics[idx].collector.dump(filename=f'{filename}_{names[idx]}', directory=filename)

    for frame in frames:
        frame['time'] -= frame['time'][0]

    draw_plots_i_y(frames, names, ['delta', 'gain', 'time', 'length'], filename, filename)
    draw_plot_x_y(frames, names, 'time', 'length', filename, filename)
