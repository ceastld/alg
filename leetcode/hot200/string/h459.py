"""
459. 重复的子字符串
给定一个非空的字符串 s ，检查是否可以通过由它的一个子串重复多次构成。

题目链接：https://leetcode.cn/problems/repeated-substring-pattern/

示例 1:
输入: s = "abab"
输出: true
解释: 可由子串 "ab" 重复两次构成。

示例 2:
输入: s = "aba"
输出: false

示例 3:
输入: s = "abcabcabcabc"
输出: true
解释: 可由子串 "abc" 重复四次构成。 (或者子串 "abcabc" 重复两次构成。)

提示：
- 1 <= s.length <= 10^4
- s 由小写英文字母组成
"""
from collections import Counter
from typing import List
import math


class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        """
        请在这里实现你的解法
        """
        n = len(s)
        count = Counter(s)
        gcd = math.gcd(*count.values())
        l0 = n // gcd
        for l in range(l0, n, l0):
            if n % l != 0:
                continue
            if s[:l] == s[-l:] and s[l:] == s[:-l]:
                return True
        return False

def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    s = "abab"
    assert solution.repeatedSubstringPattern(s) == True
    
    # 测试用例2
    s = "aba"
    assert solution.repeatedSubstringPattern(s) == False
    
    # 测试用例3
    s = "abcabcabcabc"
    assert solution.repeatedSubstringPattern(s) == True
    
    # 测试用例4
    s = "a"
    assert solution.repeatedSubstringPattern(s) == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
