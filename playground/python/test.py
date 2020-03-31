from src.graph import Graph
from src.tsp.oliver30 import tsp

graph = Graph(tsp)
print(graph.c)
graph.prim_tree()
print(graph.edges)
graph.draw()

