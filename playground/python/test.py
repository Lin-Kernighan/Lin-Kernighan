from src.algorithms.heuristics.k_opt import KOpt
from src.algorithms.heuristics.three_opt import ThreeOpt
from src.algorithms.heuristics.two_opt import TwoOpt
from src.test import save_test

save_test([KOpt, TwoOpt, ThreeOpt], ['k_opt', 'two_opt', 'three_opt'], 500)
