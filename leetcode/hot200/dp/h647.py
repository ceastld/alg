"""
647. 回文子串
给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。

回文字符串 是正着读和倒着读一样的字符串。

子字符串 是字符串中的由连续字符组成的一个序列。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

题目链接：https://leetcode.cn/problems/palindromic-substrings/

示例 1:
输入：s = "abc"
输出：3
解释：三个回文子串: "a", "b", "c"

示例 2:
输入：s = "aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"

提示：
- 1 <= s.length <= 1000
- s 由小写英文字母组成
"""

from typing import List


class Solution:
    def countSubstrings(self, s: str) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    s = "abc"
    result = solution.countSubstrings(s)
    expected = 3
    assert result == expected
    
    # 测试用例2
    s = "aaa"
    result = solution.countSubstrings(s)
    expected = 6
    assert result == expected
    
    # 测试用例3
    s = "a"
    result = solution.countSubstrings(s)
    expected = 1
    assert result == expected
    
    # 测试用例4
    s = "ab"
    result = solution.countSubstrings(s)
    expected = 2
    assert result == expected
    
    # 测试用例5
    s = "racecar"
    result = solution.countSubstrings(s)
    expected = 10
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
