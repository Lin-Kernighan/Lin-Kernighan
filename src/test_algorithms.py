import pytest

from src.algorithms.lk_opt import LKOpt
from src.algorithms.lkh_opt import LKHOpt
from src.algorithms.structures.matrix import adjacency_matrix
from src.algorithms.three_opt import ThreeOpt
from src.algorithms.two_opt import TwoOpt
from src.algorithms.utils.generator import generator
from src.algorithms.utils.initial_tour import greedy, two_opt
from src.algorithms.utils.utils import get_length
from src.lkh_search import LKHSearch
from src.tabu_proc_search import TabuProcSearch
from src.tabu_search import TabuSearch

size = 100


@pytest.fixture
def generate_metric_tsp():
    """ Генерируем данные для метрической задачи коммивояжера """
    tsp = generator(size)
    matrix = adjacency_matrix(tsp)
    length, tour = greedy(matrix)
    return length, tour, matrix


def test_greedy():
    tsp = generator(size)
    matrix = adjacency_matrix(tsp)
    length, tour = greedy(matrix)
    assert round(get_length(tour, matrix), 2) == round(length, 2), 'generated wrong tour'


def test_greedy_with_two_opt():
    tsp = generator(size)
    matrix = adjacency_matrix(tsp)
    length, tour = two_opt(matrix)
    assert round(get_length(tour, matrix), 2) == round(length, 2), 'generated wrong tour'


def test_two_opt(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    opt = TwoOpt(length, tour, matrix)
    opt_length, opt_tour = opt.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_fast_two_opt(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    opt_length, opt_tour = TwoOpt.just_improve(length, tour, matrix)
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_three_opt(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    three_opt = ThreeOpt(length, tour, matrix)
    opt_length, opt_tour = three_opt.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_fast_three_opt(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    opt_length, opt_tour = ThreeOpt.just_improve(length, tour, matrix)
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_lk_opt_simple(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lk_opt = LKOpt(length, tour, matrix, dlb=False)
    opt_length, opt_tour = lk_opt.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_lk_opt_dlb(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lk_opt = LKOpt(length, tour, matrix, dlb=True)
    opt_length, opt_tour = lk_opt.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_lk_opt_full_bridge(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lk_opt = LKOpt(length, tour, matrix, bridge=(1, True))
    opt_length, opt_tour = lk_opt.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_lkh_opt_simple(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lkh_opt = LKHOpt(length, tour, matrix, dlb=False)
    opt_length, opt_tour = lkh_opt.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_lkh_opt_dlb(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lkh_opt = LKHOpt(length, tour, matrix, dlb=True)
    opt_length, opt_tour = lkh_opt.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_lkh_opt_non_seq(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lkh_opt = LKHOpt(length, tour, matrix, bridge=True, non_seq=True)
    opt_length, opt_tour = lkh_opt.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_tabu_search_two_opt(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    tabu_search = TabuSearch('two_opt', matrix)
    opt_length, opt_tour = tabu_search.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_tabu_search_three_opt(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    tabu_search = TabuSearch('three_opt', matrix)
    opt_length, opt_tour = tabu_search.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_tabu_search_lk_opt(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    tabu_search = TabuSearch('lk_opt', matrix)
    opt_length, opt_tour = tabu_search.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_tabu_search_lkh_opt(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    tabu_search = TabuSearch('lkh_opt', matrix)
    opt_length, opt_tour = tabu_search.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_tabu_search_with_collect(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    tabu_search = TabuSearch('two_opt', matrix, collect=True)
    opt_length, opt_tour = tabu_search.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_lkh_search_fast_helsgaun(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lkh_search = LKHSearch(matrix, init='fast_helsgaun')
    opt_length, opt_tour = lkh_search.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_lkh_search_two_opt(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lkh_search = LKHSearch(matrix, init='two_opt')
    opt_length, opt_tour = lkh_search.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_lkh_search_helsgaun(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lkh_search = LKHSearch(matrix, init='helsgaun')
    opt_length, opt_tour = lkh_search.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_lkh_search_with_collect(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lkh_search = LKHSearch(matrix, collect=True)
    opt_length, opt_tour = lkh_search.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_lkh_search_with_two_opt_init(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lkh_search = LKHSearch(matrix, two_opt=True)
    opt_length, opt_tour = lkh_search.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'


def test_tabu_proc_search(generate_metric_tsp):
    length, tour, matrix = generate_metric_tsp
    lkh_search = TabuProcSearch('two_opt', matrix)
    opt_length, opt_tour = lkh_search.optimize()
    assert opt_length < length, 'optimized'
    assert round(get_length(opt_tour, matrix), 2) == round(opt_length, 2), 'generated wrong tour'
