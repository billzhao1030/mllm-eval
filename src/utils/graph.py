from collections import defaultdict, deque
import numpy as np

class NavGraph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def update_connection(self, node1, node2):
        self.add_node(node1)
        self.add_node(node2)
        if node2 in self.graph[node1]:
            return None
        self.graph[node1].append(node2)
        self.graph[node2].append(node1)

    def bfs_shortest_path(self, start, end):
        if start not in self.graph or end not in self.graph:
            return None

        visited = {start: None}
        queue = deque([start])

        while queue:
            current_node = queue.popleft()

            if current_node == end:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = visited[current_node]
                return path[::-1]

            for neighbor in self.graph[current_node]:
                if neighbor not in visited:
                    visited[neighbor] = current_node
                    queue.append(neighbor)

        return None