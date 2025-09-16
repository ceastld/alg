class Solution:
    def calcEquation(self, equations: list[list[str]], values: list[float], queries: list[list[str]]) -> list[float]:
        from collections import defaultdict, deque
        
        # 构建图
        graph = defaultdict(dict)
        for (a, b), val in zip(equations, values):
            graph[a][b] = val
            graph[b][a] = 1 / val
        
        def bfs(start, end):
            if start not in graph or end not in graph:
                return -1.0
            
            if start == end:
                return 1.0
            
            queue = deque([(start, 1.0)])
            visited = {start}
            
            while queue:
                node, value = queue.popleft()
                
                for neighbor, weight in graph[node].items():
                    if neighbor == end:
                        return value * weight
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, value * weight))
            
            return -1.0
        
        return [bfs(a, b) for a, b in queries]
