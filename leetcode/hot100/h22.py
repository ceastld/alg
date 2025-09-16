"""
LeetCode 22. Generate Parentheses

题目描述：
数字n代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且有效的括号组合。

示例：
n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]

数据范围：
- 1 <= n <= 8
"""

class Solution:
    def generateParenthesis(self, n: int) -> list[str]:
        result = []
        
        def backtrack(current, open_count, close_count):
            if len(current) == 2 * n:
                result.append(current)
                return
            
            if open_count < n:
                backtrack(current + "(", open_count + 1, close_count)
            
            if close_count < open_count:
                backtrack(current + ")", open_count, close_count + 1)
        
        backtrack("", 0, 0)
        return result
