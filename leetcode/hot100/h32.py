"""
LeetCode 32. Longest Valid Parentheses

题目描述：
给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

示例：
s = "(()"
输出：2
解释：最长有效括号子串是 "()"

数据范围：
- 0 <= s.length <= 3 * 10^4
- s[i] 为 '(' 或 ')'
"""

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]  # 栈底放-1作为边界
        max_length = 0
        
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)  # 新的边界
                else:
                    max_length = max(max_length, i - stack[-1])
        
        return max_length
