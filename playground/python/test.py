from src.graph import Graph
from src.tsp.ulysses16 import tsp

graph = Graph(tsp)
total_price = graph.prim_tree()
print(total_price)
print(graph.edges)
graph.draw()

