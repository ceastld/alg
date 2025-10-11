"""
541. 反转字符串II
给定一个字符串 s 和一个整数 k，从字符串开头算起，每计数至 2k 个字符，就反转这 2k 个字符中的前 k 个字符。

如果剩余字符少于 k 个，则将剩余字符全部反转。
如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。

题目链接：https://leetcode.cn/problems/reverse-string-ii/

示例 1:
输入: s = "abcdefg", k = 2
输出: "bacdfeg"

示例 2:
输入: s = "abcd", k = 2
输出: "bacd"

提示：
- 1 <= s.length <= 10^4
- s 只由小写英文字母组成
- 1 <= k <= 10^4
"""

from typing import List


class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        """
        请在这里实现你的解法
        """
        return "".join([s[i : i + k][::-1] + s[i + k : i + k * 2] for i in range(0, len(s), k * 2)])


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    s = "abcdefg"
    k = 2
    assert solution.reverseStr(s, k) == "bacdfeg"

    # 测试用例2
    s = "abcd"
    k = 2
    assert solution.reverseStr(s, k) == "bacd"

    # 测试用例3
    s = "a"
    k = 1
    assert solution.reverseStr(s, k) == "a"

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
