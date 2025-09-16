"""
LeetCode 322. Coin Change

题目描述：
给你一个整数数组coins，表示不同面额的硬币；以及一个整数amount，表示总金额。
计算并返回可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回-1。
你可以认为每种硬币的数量是无限的。

示例：
coins = [1,3,4], amount = 6
输出：2
解释：6 = 3 + 3

数据范围：
- 1 <= coins.length <= 12
- 1 <= coins[i] <= 2^31 - 1
- 0 <= amount <= 10^4
"""

class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        # dp[i] 表示凑成金额 i 所需的最少硬币数
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
