"""
LeetCode 64. Minimum Path Sum

题目描述：
给定一个包含非负整数的m x n网格grid，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
说明：每次只能向下或者向右移动一步。

示例：
grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径1→3→1→1→1的总和最小。

数据范围：
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 200
- 0 <= grid[i][j] <= 100
"""

class Solution:
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        
        # dp[i][j] 表示从(0,0)到(i,j)的最小路径和
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = grid[0][0]
        
        # 初始化第一行
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        
        # 初始化第一列
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        
        # 填充dp表
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        
        return dp[m-1][n-1]
