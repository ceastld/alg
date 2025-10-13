"""
518. 零钱兑换II
给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。

请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。

假设每一种面额的硬币有无限个。

题目链接：https://leetcode.cn/problems/coin-change-ii/

示例 1:
输入：amount = 5, coins = [1, 2, 5]
输出：4
解释：有四种方式可以凑成总金额：
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1

示例 2:
输入：amount = 3, coins = [2]
输出：0
解释：只用面额 2 的硬币不能凑成总金额 3 。

示例 3:
输入：amount = 10, coins = [10]
输出：1

提示：
- 1 <= coins.length <= 300
- 1 <= coins[i] <= 5000
- coins 中的所有值 互不相同
- 0 <= amount <= 5000
"""

from typing import List


class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        sum0 = 0
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        return dp[amount]

def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    amount = 5
    coins = [1, 2, 5]
    result = solution.change(amount, coins)
    expected = 4
    assert result == expected
    
    # 测试用例2
    amount = 3
    coins = [2]
    result = solution.change(amount, coins)
    expected = 0
    assert result == expected
    
    # 测试用例3
    amount = 10
    coins = [10]
    result = solution.change(amount, coins)
    expected = 1
    assert result == expected
    
    # 测试用例4
    amount = 0
    coins = [1, 2, 5]
    result = solution.change(amount, coins)
    expected = 1
    assert result == expected
    
    # 测试用例5
    amount = 5
    coins = [1, 2, 3]
    result = solution.change(amount, coins)
    expected = 5
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
