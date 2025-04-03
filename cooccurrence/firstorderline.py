import numpy as np
import networkx as nx
import random

class FirstOrderLINE:
    def __init__(self, graph, embedding_size=256, negative_ratio=5):
        self.graph = graph
        self.node_indices = list(graph.nodes())
        self.node_count = len(graph.nodes())
        self.embedding_size = embedding_size
        self.negative_ratio = negative_ratio
        self.node_embeddings = np.random.rand(self.node_count, self.embedding_size)
        self.node_embeddings_dict = {}

    def generate_samples(self):
        samples = []
        for edge in self.graph.edges():
            samples.append(edge)
        return samples

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, num_epochs=1, learning_rate=0.01):
        samples = self.generate_samples()
        for epoch in range(num_epochs):
            random.shuffle(samples)
            for node1, node2 in samples:
                positive_samples = [(self.node_indices.index(node1), self.node_indices.index(node2))]
                negative_samples = []
                for _ in range(self.negative_ratio):
                    while True:
                        node3, node4 = random.sample(self.graph.nodes(), 2)
                        if not self.graph.has_edge(node3, node4):
                            negative_samples.append((self.node_indices.index(node3), self.node_indices.index(node4)))
                            break
                for u, v in positive_samples:
                    diff = self.node_embeddings[u] - self.node_embeddings[v]
                    loss = -np.log(self.sigmoid(np.dot(diff, diff.T)))
                    grad = 2 * diff * (1 - self.sigmoid(np.dot(diff, diff.T)))
                    self.node_embeddings[u] -= learning_rate * grad
                    self.node_embeddings[v] += learning_rate * grad
                for u, v in negative_samples:
                    diff = self.node_embeddings[u] - self.node_embeddings[v]
                    loss += np.log(self.sigmoid(-np.dot(diff, diff.T)))
                    grad = -2 * diff * self.sigmoid(-np.dot(diff, diff.T))
                    self.node_embeddings[u] -= learning_rate * grad
                    self.node_embeddings[v] += learning_rate * grad
            learning_rate *= 0.95
        for i, row in enumerate(self.node_embeddings):
            if len(row) == 0:
                self.node_embeddings_dict[i] = np.zeros(self.embedding_size,dtype=np.float32)
            else:
                self.node_embeddings_dict[i] = np.array(row,dtype=np.float32)