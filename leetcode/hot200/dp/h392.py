"""
392. 判断子序列
给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

题目链接：https://leetcode.cn/problems/is-subsequence/

示例 1:
输入：s = "abc", t = "ahbgdc"
输出：true

示例 2:
输入：s = "axc", t = "ahbgdc"
输出：false

提示：
- 0 <= s.length <= 100
- 0 <= t.length <= 10^4
- 两个字符串都只由小写字符组成
"""

from typing import List


class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        j = 0
        if not s:
            return True
        if not t:
            return False
        for i in range(len(t)):
            if t[i] == s[j]:
                j += 1
            if j == len(s):
                return True
        return False


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    s = "abc"
    t = "ahbgdc"
    result = solution.isSubsequence(s, t)
    expected = True
    assert result == expected

    # 测试用例2
    s = "axc"
    t = "ahbgdc"
    result = solution.isSubsequence(s, t)
    expected = False
    assert result == expected

    # 测试用例3
    s = ""
    t = "ahbgdc"
    result = solution.isSubsequence(s, t)
    expected = True
    assert result == expected

    # 测试用例4
    s = "abc"
    t = ""
    result = solution.isSubsequence(s, t)
    expected = False
    assert result == expected

    # 测试用例5
    s = "a"
    t = "a"
    result = solution.isSubsequence(s, t)
    expected = True
    assert result == expected

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
