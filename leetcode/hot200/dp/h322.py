"""
322. 零钱兑换
给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。

题目链接：https://leetcode.cn/problems/coin-change/

示例 1:
输入：coins = [1, 3, 4], amount = 6
输出：2
解释：6 = 3 + 3

示例 2:
输入：coins = [2], amount = 3
输出：-1

示例 3:
输入：coins = [1], amount = 0
输出：0

提示：
- 1 <= coins.length <= 12
- 1 <= coins[i] <= 2^31 - 1
- 0 <= amount <= 10^4
"""

from typing import List


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1

def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    coins = [1, 3, 4]
    amount = 6
    result = solution.coinChange(coins, amount)
    expected = 2
    assert result == expected
    
    # 测试用例2
    coins = [2]
    amount = 3
    result = solution.coinChange(coins, amount)
    expected = -1
    assert result == expected
    
    # 测试用例3
    coins = [1]
    amount = 0
    result = solution.coinChange(coins, amount)
    expected = 0
    assert result == expected
    
    # 测试用例4
    coins = [1, 2, 5]
    amount = 11
    result = solution.coinChange(coins, amount)
    expected = 3
    assert result == expected
    
    # 测试用例5
    coins = [2, 5, 10, 1]
    amount = 27
    result = solution.coinChange(coins, amount)
    expected = 4
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
