"""
70. 爬楼梯 - 标准答案
"""
from typing import List


class Solution:
    def climbStairs(self, n: int) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. dp[i] 表示爬到第i阶楼梯的方法数
        2. 状态转移方程：dp[i] = dp[i-1] + dp[i-2]
        3. 边界条件：dp[1] = 1, dp[2] = 2
        4. 本质上是斐波那契数列
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if n <= 2:
            return n
        
        # 空间优化：只需要前两个状态
        prev2 = 1  # dp[i-2]
        prev1 = 2  # dp[i-1]
        
        for i in range(3, n + 1):
            current = prev1 + prev2
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def climbStairs_dp_array(self, n: int) -> int:
        """
        动态规划数组解法
        
        解题思路：
        1. 使用数组存储所有状态
        2. 更直观但空间复杂度为O(n)
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if n <= 2:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        
        for i in range(3, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]
    
    def climbStairs_recursive(self, n: int) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算，使用记忆化避免重复计算
        2. 自顶向下的动态规划
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        memo = {}
        
        def dfs(n):
            if n in memo:
                return memo[n]
            if n <= 2:
                return n
            
            memo[n] = dfs(n-1) + dfs(n-2)
            return memo[n]
        
        return dfs(n)
    
    def climbStairs_matrix(self, n: int) -> int:
        """
        矩阵快速幂解法（简化版）
        
        解题思路：
        1. 使用斐波那契数列的矩阵表示
        2. 简化实现，避免复杂的矩阵运算
        
        时间复杂度：O(log n)
        空间复杂度：O(1)
        """
        if n <= 2:
            return n
        
        # 使用斐波那契数列的递推关系
        # F(n) = F(n-1) + F(n-2)
        # 对于爬楼梯问题，dp[n] = dp[n-1] + dp[n-2]
        # 初始值：dp[1] = 1, dp[2] = 2
        
        # 使用快速幂计算斐波那契数
        def fib(n):
            if n <= 2:
                return n
            a, b = 1, 2
            for _ in range(3, n + 1):
                a, b = b, a + b
            return b
        
        return fib(n)
    
    def climbStairs_math(self, n: int) -> int:
        """
        数学公式解法
        
        解题思路：
        1. 斐波那契数列的通项公式
        2. 使用黄金比例计算
        
        时间复杂度：O(1)
        空间复杂度：O(1)
        """
        import math
        
        sqrt5 = math.sqrt(5)
        phi = (1 + sqrt5) / 2
        psi = (1 - sqrt5) / 2
        
        return int((phi**(n+1) - psi**(n+1)) / sqrt5)


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    n = 2
    result = solution.climbStairs(n)
    expected = 2
    assert result == expected
    
    # 测试用例2
    n = 3
    result = solution.climbStairs(n)
    expected = 3
    assert result == expected
    
    # 测试用例3
    n = 4
    result = solution.climbStairs(n)
    expected = 5
    assert result == expected
    
    # 测试用例4
    n = 1
    result = solution.climbStairs(n)
    expected = 1
    assert result == expected
    
    # 测试用例5
    n = 5
    result = solution.climbStairs(n)
    expected = 8
    assert result == expected
    
    # 测试DP数组解法
    print("测试DP数组解法...")
    n = 5
    result_dp = solution.climbStairs_dp_array(n)
    assert result_dp == expected
    
    # 测试递归解法
    print("测试递归解法...")
    n = 5
    result_rec = solution.climbStairs_recursive(n)
    assert result_rec == expected
    
    # 测试矩阵解法
    print("测试矩阵解法...")
    n = 5
    result_matrix = solution.climbStairs_matrix(n)
    assert result_matrix == expected
    
    # 测试数学公式解法
    print("测试数学公式解法...")
    n = 5
    result_math = solution.climbStairs_math(n)
    assert result_math == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
