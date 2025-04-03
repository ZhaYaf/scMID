import networkx as nx
import random
import numpy as np


def random_choice(graph):
    nodes = list(graph.nodes)
    index = random.randint(0, len(nodes)-1)
    return nodes[index]

def random_walk(G,max_rounds=10000, max_steps=10000):
    node_counts = {node: 0 for node in G.nodes()}
    walks = []
    for _ in range(max_rounds):
        start_node = random_choice(G)
        walk = [start_node]
        current_node = start_node
        for _ in range(max_steps):
            node_counts[current_node] += 1
            neighbors = list(G.neighbors(current_node))
            if neighbors:
                current_node = random.choice(neighbors)
            else:
                break
    node_embeddings = {}
    return node_embeddings,node_counts
