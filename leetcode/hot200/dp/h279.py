"""
279. 完全平方数
给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

题目链接：https://leetcode.cn/problems/perfect-squares/

示例 1:
输入：n = 12
输出：3
解释：12 = 4 + 4 + 4

示例 2:
输入：n = 13
输出：2
解释：13 = 4 + 9

提示：
- 1 <= n <= 10^4
"""

from math import sqrt
from typing import List


class Solution:
    def numSquares(self, n: int) -> int:
        # 贪心不行
        dp = [float('inf')] * (n + 1)
        for i in range(1, int(sqrt(n)) + 1):
            dp[i * i] = 1
        for i in range(1, n + 1):
            for j in range(1, int(sqrt(i)) + 1):
                dp[i] = min(dp[i], dp[i - j * j] + 1)
        print(dp)
        return dp[n]


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    n = 12
    result = solution.numSquares(n)
    expected = 3
    assert result == expected
    
    # 测试用例2
    n = 13
    result = solution.numSquares(n)
    expected = 2
    assert result == expected
    
    # 测试用例3
    n = 1
    result = solution.numSquares(n)
    expected = 1
    assert result == expected
    
    # 测试用例4
    n = 4
    result = solution.numSquares(n)
    expected = 1
    assert result == expected
    
    # 测试用例5
    n = 9
    result = solution.numSquares(n)
    expected = 1
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
