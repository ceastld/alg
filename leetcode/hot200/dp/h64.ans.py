"""
64. 最小路径和 - 标准答案
"""
from typing import List


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        标准解法：二维动态规划
        
        解题思路：
        1. dp[i][j] 表示从(0,0)到(i,j)的最小路径和
        2. 状态转移方程：dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        3. 边界条件：第一行和第一列只能从左边或上边来
        4. 空间优化：可以优化到O(n)空间
        
        时间复杂度：O(m×n)
        空间复杂度：O(m×n)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        
        # 初始化起点
        dp[0][0] = grid[0][0]
        
        # 初始化第一行
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        
        # 初始化第一列
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        
        # 填充其余位置
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        
        return dp[m-1][n-1]
    
    def minPathSum_optimized(self, grid: List[List[int]]) -> int:
        """
        空间优化解法：滚动数组
        
        解题思路：
        1. 使用一维数组代替二维数组
        2. 每次只保存当前行的状态
        3. 空间复杂度从O(m×n)优化到O(n)
        
        时间复杂度：O(m×n)
        空间复杂度：O(n)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        dp = [0] * n
        
        # 初始化第一行
        dp[0] = grid[0][0]
        for j in range(1, n):
            dp[j] = dp[j-1] + grid[0][j]
        
        # 逐行处理
        for i in range(1, m):
            # 更新第一列
            dp[0] += grid[i][0]
            
            # 更新其余列
            for j in range(1, n):
                dp[j] = min(dp[j], dp[j-1]) + grid[i][j]
        
        return dp[n-1]
    
    def minPathSum_inplace(self, grid: List[List[int]]) -> int:
        """
        原地修改解法：直接修改原数组
        
        解题思路：
        1. 直接在原数组上进行状态转移
        2. 不需要额外的空间
        3. 修改原数组，节省空间
        
        时间复杂度：O(m×n)
        空间复杂度：O(1)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        # 初始化第一行
        for j in range(1, n):
            grid[0][j] += grid[0][j-1]
        
        # 初始化第一列
        for i in range(1, m):
            grid[i][0] += grid[i-1][0]
        
        # 填充其余位置
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        
        return grid[m-1][n-1]
    
    def minPathSum_recursive(self, grid: List[List[int]]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算，使用记忆化避免重复计算
        2. 自顶向下的动态规划
        
        时间复杂度：O(m×n)
        空间复杂度：O(m×n)
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        memo = {}
        
        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i == 0 and j == 0:
                return grid[0][0]
            
            if i < 0 or j < 0:
                return float('inf')
            
            memo[(i, j)] = min(dfs(i-1, j), dfs(i, j-1)) + grid[i][j]
            return memo[(i, j)]
        
        return dfs(m-1, n-1)
    
    def minPathSum_bfs(self, grid: List[List[int]]) -> int:
        """
        BFS解法：广度优先搜索
        
        解题思路：
        1. 将问题转化为图的最短路径问题
        2. 每个格子是一个节点，相邻格子有边
        3. 使用BFS找到最短路径
        
        时间复杂度：O(m×n)
        空间复杂度：O(m×n)
        """
        if not grid or not grid[0]:
            return 0
        
        from collections import deque
        
        m, n = len(grid), len(grid[0])
        queue = deque([(0, 0, grid[0][0])])
        visited = {(0, 0): grid[0][0]}
        
        while queue:
            i, j, cost = queue.popleft()
            
            if i == m-1 and j == n-1:
                return cost
            
            # 向右移动
            if j + 1 < n:
                new_cost = cost + grid[i][j+1]
                if (i, j+1) not in visited or new_cost < visited[(i, j+1)]:
                    visited[(i, j+1)] = new_cost
                    queue.append((i, j+1, new_cost))
            
            # 向下移动
            if i + 1 < m:
                new_cost = cost + grid[i+1][j]
                if (i+1, j) not in visited or new_cost < visited[(i+1, j)]:
                    visited[(i+1, j)] = new_cost
                    queue.append((i+1, j, new_cost))
        
        return -1


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    grid = [[1,3,1],[1,5,1],[4,2,1]]
    result = solution.minPathSum(grid)
    expected = 7
    assert result == expected
    
    # 测试用例2
    grid = [[1,2,3],[4,5,6]]
    result = solution.minPathSum(grid)
    expected = 12
    assert result == expected
    
    # 测试用例3
    grid = [[1]]
    result = solution.minPathSum(grid)
    expected = 1
    assert result == expected
    
    # 测试用例4
    grid = [[1,2],[1,1]]
    result = solution.minPathSum(grid)
    expected = 3
    assert result == expected
    
    # 测试空间优化解法
    print("测试空间优化解法...")
    grid = [[1,3,1],[1,5,1],[4,2,1]]
    result_opt = solution.minPathSum_optimized(grid)
    expected_opt = 7
    assert result_opt == expected_opt
    
    # 测试原地修改解法
    print("测试原地修改解法...")
    grid = [[1,3,1],[1,5,1],[4,2,1]]
    result_inplace = solution.minPathSum_inplace(grid)
    expected_inplace = 7
    assert result_inplace == expected_inplace
    
    # 测试递归解法
    print("测试递归解法...")
    grid = [[1,3,1],[1,5,1],[4,2,1]]
    result_rec = solution.minPathSum_recursive(grid)
    expected_rec = 7
    assert result_rec == expected_rec
    
    # 测试BFS解法
    print("测试BFS解法...")
    grid = [[1,3,1],[1,5,1],[4,2,1]]
    result_bfs = solution.minPathSum_bfs(grid)
    expected_bfs = 7
    assert result_bfs == expected_bfs
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
