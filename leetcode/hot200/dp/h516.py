"""
516. 最长回文子序列
给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。

题目链接：https://leetcode.cn/problems/longest-palindromic-subsequence/

示例 1:
输入：s = "bbbab"
输出：4
解释：一个可能的最长回文子序列为 "bbbb"。

示例 2:
输入：s = "cbbd"
输出：2
解释：一个可能的最长回文子序列为 "bb"。

提示：
- 1 <= s.length <= 1000
- s 仅由小写英文字母组成
"""

from typing import List


class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        return self.longestCommonSubsequence(s, s[::-1])
        
    def longestCommonSubsequence(self, s1: str, s2: str) -> int:
        m,n = len(s1),len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    s = "bbbab"
    result = solution.longestPalindromeSubseq(s)
    expected = 4
    assert result == expected
    
    # 测试用例2
    s = "cbbd"
    result = solution.longestPalindromeSubseq(s)
    expected = 2
    assert result == expected
    
    # 测试用例3
    s = "a"
    result = solution.longestPalindromeSubseq(s)
    expected = 1
    assert result == expected
    
    # 测试用例4
    s = "ab"
    result = solution.longestPalindromeSubseq(s)
    expected = 1
    assert result == expected
    
    # 测试用例5
    s = "racecar"
    result = solution.longestPalindromeSubseq(s)
    expected = 7
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
