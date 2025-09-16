"""
LeetCode 70. Climbing Stairs

题目描述：
假设你正在爬楼梯。需要n阶你才能到达楼顶。
每次你可以爬1或2个台阶。你有多少种不同的方法可以爬到楼顶呢？

示例：
n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1阶 + 1阶
2. 2阶

数据范围：
- 1 <= n <= 45
"""

class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return n
        
        # 动态规划：dp[i]表示爬到第i阶的方法数
        dp = [0] * (n + 1)
        dp[1] = 1  # 爬到第1阶有1种方法
        dp[2] = 2  # 爬到第2阶有2种方法
        
        for i in range(3, n + 1):
            # 爬到第i阶 = 从第i-1阶爬1步 + 从第i-2阶爬2步
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]
