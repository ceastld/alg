"""
LeetCode 10. Regular Expression Matching

题目描述：
给你一个字符串s和一个字符规律p，请你来实现一个支持'.'和'*'的正则表达式匹配。
'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素

示例：
s = "aa", p = "a*"
输出：true
解释：'*' 表示可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'

数据范围：
- 1 <= s.length <= 20
- 1 <= p.length <= 30
- s 只包含从 a-z 的小写字母
- p 只包含从 a-z 的小写字母，以及字符 . 和 *
- 保证每次出现字符 * 时，前面都匹配到有效的字符
"""

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        # 处理模式串开头的 * 情况
        for j in range(2, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    # * 匹配0个字符
                    dp[i][j] = dp[i][j - 2]
                    # * 匹配1个或多个字符
                    if self.match(s[i - 1], p[j - 2]):
                        dp[i][j] = dp[i][j] or dp[i - 1][j]
                else:
                    # 普通字符匹配
                    if self.match(s[i - 1], p[j - 1]):
                        dp[i][j] = dp[i - 1][j - 1]
        
        return dp[m][n]
    
    def match(self, a: str, b: str) -> bool:
        return a == b or b == '.'
