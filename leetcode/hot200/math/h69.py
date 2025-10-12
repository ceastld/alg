"""
69. x 的平方根
给你一个非负整数 x ，计算并返回 x 的 算术平方根 。

由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。

注意：不允许使用任何内置指数函数和算符，比如 pow(x, 0.5) 或者 x ** 0.5 。

题目链接：https://leetcode.cn/problems/sqrtx/

示例 1:
输入：x = 4
输出：2

示例 2:
输入：x = 8
输出：2
解释：8 的算术平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。

提示：
- 0 <= x <= 2^31 - 1
"""

from typing import List



class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        if x < 4:
            return 1

        left, right = 0, x
        while left < right - 1:
            mid = (left + right) >> 1
            if mid * mid <= x:
                left = mid
            else:
                right = mid
        return left


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    x = 196
    result = solution.mySqrt(x)
    expected = 14
    assert result == expected

    # 测试用例2
    x = 8
    result = solution.mySqrt(x)
    expected = 2
    assert result == expected

    # 测试用例3
    x = 0
    result = solution.mySqrt(x)
    expected = 0
    assert result == expected

    # 测试用例4
    x = 1
    result = solution.mySqrt(x)
    expected = 1
    assert result == expected

    # 测试用例5
    x = 9
    result = solution.mySqrt(x)
    expected = 3
    assert result == expected

    # 测试用例6
    x = 15
    result = solution.mySqrt(x)
    expected = 3
    assert result == expected

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
