"""
LeetCode 5. Longest Palindromic Substring

题目描述：
给你一个字符串s，找到s中最长的回文子串。

示例：
s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案

数据范围：
- 1 <= s.length <= 1000
- s 仅由数字和英文字母组成
"""

class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""
        
        start = end = 0
        
        for i in range(len(s)):
            # 奇数长度回文串
            len1 = self.expandAroundCenter(s, i, i)
            # 偶数长度回文串
            len2 = self.expandAroundCenter(s, i, i + 1)
            
            max_len = max(len1, len2)
            
            if max_len > end - start:
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
        
        return s[start:end + 1]
    
    def expandAroundCenter(self, s: str, left: int, right: int) -> int:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
