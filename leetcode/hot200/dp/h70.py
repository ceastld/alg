"""
70. 爬楼梯
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

题目链接：https://leetcode.cn/problems/climbing-stairs/

示例 1:
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶

示例 2:
输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶

提示：
- 1 <= n <= 45
"""

from typing import List


class Solution:
    def climbStairs(self, n: int) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
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
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
