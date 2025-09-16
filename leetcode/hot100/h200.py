"""
LeetCode 200. Number of Islands

题目描述：
给你一个由'1'（陆地）和'0'（水）组成的二维网格，请你计算网格中岛屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
此外，你可以假设该网格的四条边均被水包围。

示例：
grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1

数据范围：
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 300
- grid[i][j]的值为'0'或'1'
"""

from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """计算岛屿数量 - 简化版本"""
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        
        def dfs(r: int, c: int) -> None:
            """DFS遍历岛屿"""
            if (r < 0 or r >= rows or c < 0 or c >= cols or 
                grid[r][c] != "1"):
                return
            
            grid[r][c] = "0"  # 标记为已访问
            
            # 遍历四个方向
            for dr, dc in directions:
                dfs(r + dr, c + dc)
        
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1":
                    dfs(r, c)
                    count += 1
        
        return count