"""
LeetCode 301. Remove Invalid Parentheses

题目描述：
给你一个由若干括号和字母组成的字符串s，删除最小数量的无效括号，使得输入的字符串有效。
返回所有可能的结果。答案可以按任意顺序返回。

示例：
s = "()())()"
输出：["(())()","()()()"]

数据范围：
- 1 <= s.length <= 25
- s由小写英文字母以及括号'('和')'组成
- s中至多含20个括号
"""

class Solution:
    def removeInvalidParentheses(self, s: str) -> list[str]:
        def is_valid(s):
            count = 0
            for char in s:
                if char == '(':
                    count += 1
                elif char == ')':
                    count -= 1
                    if count < 0:
                        return False
            return count == 0
        
        def dfs(s, start, left_remove, right_remove):
            if left_remove == 0 and right_remove == 0:
                if is_valid(s):
                    result.add(s)
                return
            
            for i in range(start, len(s)):
                if i > start and s[i] == s[i-1]:
                    continue
                
                if s[i] == '(' and left_remove > 0:
                    dfs(s[:i] + s[i+1:], i, left_remove - 1, right_remove)
                if s[i] == ')' and right_remove > 0:
                    dfs(s[:i] + s[i+1:], i, left_remove, right_remove - 1)
        
        # 计算需要删除的左右括号数量
        left_remove = right_remove = 0
        for char in s:
            if char == '(':
                left_remove += 1
            elif char == ')':
                if left_remove > 0:
                    left_remove -= 1
                else:
                    right_remove += 1
        
        result = set()
        dfs(s, 0, left_remove, right_remove)
        return list(result)
