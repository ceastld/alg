"""
LeetCode 399. Evaluate Division

题目描述：
给你一个变量对数组equations和一个实数值数组values作为已知条件，其中equations[i] = [Ai, Bi]和values[i]共同表示等式Ai / Bi = values[i]。每个Ai或Bi是一个表示单个变量的字符串。
另有一些以数组queries表示的问题，其中queries[j] = [Cj, Dj]表示第j个问题，请你根据已知条件找出Cj / Dj = ?的结果作为答案。
返回所有问题的答案。如果存在某个无法确定的答案，则用-1.0替代这个答案。

示例：
equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
输出：[6.00000,0.50000,-1.00000,1.00000,-1.00000]

数据范围：
- 1 <= equations.length <= 20
- equations[i].length == 2
- 1 <= Ai.length, Bi.length <= 5
- values.length == equations.length
- 0.0 < values[i] <= 20.0
- 1 <= queries.length <= 20
- queries[i].length == 2
- 1 <= Cj.length, Dj.length <= 5
- Ai, Bi, Cj, Dj由小写英文字母与数字组成
"""

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
