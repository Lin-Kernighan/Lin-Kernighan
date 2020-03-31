from src.graph import Graph
from src.tsp.ulysses16 import tsp

graph = Graph(tsp)
graph.prim_tree()
print(graph.edges)
graph.draw()

