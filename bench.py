import logging
from time import time

import click

from src.algorithms.structures.matrix import adjacency_matrix
from src.algorithms.utils.generator import generator
from src.algorithms.utils.initial_tour import greedy
from src.lkh_search import LKHSearch
from src.tabu_proc_search import TabuProcSearch
from src.tabu_search import TabuSearch
from src.utils import opts_type

search_type = dict(lkh=LKHSearch, tabu=TabuSearch, tabu_p=TabuProcSearch)


@click.group()
def cli():
    pass


@click.command()
@click.option('--search', default='lkh', help='Search can be: lkh, tabu, tabu_p')
@click.option('--size', default=100, help='Size of generated tsp task')
@click.option('--number', default=5, help='Number of launching for averaging')
@click.option('--info', default=False, help='Show info message')
@click.option('--opt', default='two_opt', help='Opt type for tabu searchers: two_opt, three_opt, lk_opt, lkh_opt')
@click.option('--iterations', default=10, help='Iterations of search')
@click.option('--swap', default=2, help='Swaps for tabu searchers')
@click.option('--proc', default=4, help='Number of process in tabu_p')
def searchers(search, size, number, info, opt, iterations, swap, proc):
    if info:
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    average = 0.
    ft_start = time()
    for _ in range(number):
        tsp = generator(size)
        matrix = adjacency_matrix(tsp)

        t_start = time()
        searcher = search_type[search](matrix=matrix, opt=opt, proc=proc)
        searcher.optimize(iterations=iterations, swap=swap)
        average += (time() - t_start)

    click.echo(f'all: {time() - ft_start} s')
    click.echo(f'average: {average / number} s')


@click.command()
@click.option('--opt', default='two_opt', help='Opt type: two_opt, three_opt, lk_opt, lkh_opt')
@click.option('--size', default=100, help='Size of generated tsp task')
@click.option('--number', default=5, help='Number of launching for averaging')
@click.option('--info', default=False, help='Show info message')
@click.option('--neighbours', default=5, help='Neighbours for LK')
@click.option('--k', default=5, help='Number of k for LK or LKH')
@click.option('--mul', default=1, help='Excess factor for LKH (factor * excess)')
@click.option('--sb', default=False, help='Use or not subgradient optimization ofr LKH')
def opts(opt, size, number, info, neighbours, k, mul, sb):
    if info:
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    average = 0.
    ft_start = time()
    for _ in range(number):
        tsp = generator(size)
        matrix = adjacency_matrix(tsp)
        length, tour = greedy(matrix)

        t_start = time()
        searcher = opts_type[opt](length, tour, matrix, neighbours=neighbours, k=k, mul=mul, subgradient=sb)
        searcher.optimize()
        average += (time() - t_start)

    click.echo(f'all: {time() - ft_start} s')
    click.echo(f'average: {average / number} s')


cli.add_command(searchers)
cli.add_command(opts)

if __name__ == '__main__':
    cli()
