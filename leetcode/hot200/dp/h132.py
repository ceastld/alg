"""
132. 分割回文串II
给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文。

返回符合要求的 最少分割次数 。

题目链接：https://leetcode.cn/problems/palindrome-partitioning-ii/

示例 1:
输入：s = "aab"
输出：1
解释：只需一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。

示例 2:
输入：s = "a"
输出：0

示例 3:
输入：s = "ab"
输出：1

提示：
- 1 <= s.length <= 2000
- s 仅由小写英文字母组成
"""

from typing import List


class Solution:
    def minCut(self, s: str) -> int:
        # is_palindrome[l][r] 表示 s[l:r+1] 是否为回文串
        n = len(s)
        is_palindrome = [[True] * n for _ in range(n)]
        for l in range(n - 2, -1, -1):
            for r in range(l + 1, n):
                is_palindrome[l][r] = s[l] == s[r] and is_palindrome[l + 1][r - 1]
        dp = [float('inf')] * (len(s) + 1)
        dp[0] = 0
        for i in range(1, len(s) + 1):
            for j in range(i):
                if is_palindrome[j][i-1]:
                    dp[i] = min(dp[i], dp[j] + 1)
        print(dp)
        return dp[len(s)]-1


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    s = "aab"
    result = solution.minCut(s)
    expected = 1
    assert result == expected
    
    # 测试用例2
    s = "a"
    result = solution.minCut(s)
    expected = 0
    assert result == expected
    
    # 测试用例3
    s = "ab"
    result = solution.minCut(s)
    expected = 1
    assert result == expected
    
    # 测试用例4
    s = "racecar"
    result = solution.minCut(s)
    expected = 0
    assert result == expected
    
    # 测试用例5
    s = "abacaba"
    result = solution.minCut(s)
    expected = 0
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
