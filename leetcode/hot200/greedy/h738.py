"""
738. 单调递增的数字
当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的。

给定一个整数 n ，返回 小于或等于 n 的最大数字，且数字呈单调递增。

题目链接：https://leetcode.cn/problems/monotone-increasing-digits/

示例 1:
输入: n = 10
输出: 9

示例 2:
输入: n = 1234
输出: 1234

示例 3:
输入: n = 332
输出: 299

提示：
- 0 <= n <= 10^9
"""

class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 将数字转换为字符串
        2. 从右到左遍历，找到第一个不满足单调递增的位置
        3. 将该位置减1，并将后面的所有位置设为9
        4. 贪心策略：尽可能保持高位数字不变
        
        时间复杂度：O(log n)
        空间复杂度：O(log n)
        """
        # 将数字转换为字符串
        s = list(str(n))
        
        # 从右到左遍历，找到第一个不满足单调递增的位置
        for i in range(len(s) - 1, 0, -1):
            if s[i] < s[i - 1]:
                # 将前一位减1，并将后面的所有位置设为9
                s[i - 1] = str(int(s[i - 1]) - 1)
                for j in range(i, len(s)):
                    s[j] = '9'
        
        return int(''.join(s))


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    assert solution.monotoneIncreasingDigits(10) == 9

    # 测试用例2
    assert solution.monotoneIncreasingDigits(1234) == 1234

    # 测试用例3
    assert solution.monotoneIncreasingDigits(332) == 299

    # 测试用例4
    assert solution.monotoneIncreasingDigits(120) == 119

    # 测试用例5
    assert solution.monotoneIncreasingDigits(100) == 99

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
