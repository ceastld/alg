"""
202. 快乐数
编写一个算法来判断一个数 n 是不是快乐数。

「快乐数」 定义为：
- 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
- 然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
- 如果 这个过程 结果为 1，那么这个数就是快乐数。

如果 n 是 快乐数就返回 true ；不是，则返回 false 。

题目链接：https://leetcode.cn/problems/happy-number/

示例 1:
输入：n = 19
输出：true
解释：
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1

示例 2:
输入：n = 2
输出：false

提示：
- 1 <= n <= 231 - 1
"""


class Solution:
    """
    202. 快乐数
    哈希表经典题目
    """

    def isHappy(self, n: int) -> bool:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        seen = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = self.f(n)
        return n == 1

    def f(self, n: int) -> int:
        total = 0
        while n > 0:
            digit = n % 10
            total += digit * digit
            n //= 10
        return total


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    assert solution.isHappy(19) == True

    # 测试用例2
    assert solution.isHappy(2) == False

    # 测试用例3
    assert solution.isHappy(1) == True

    # 测试用例4
    assert solution.isHappy(7) == True

    # 测试用例5
    assert solution.isHappy(4) == False

    # 测试用例6
    assert solution.isHappy(10) == True

    # 测试用例7
    assert solution.isHappy(13) == True

    # 测试用例8
    assert solution.isHappy(16) == False

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
