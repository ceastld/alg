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