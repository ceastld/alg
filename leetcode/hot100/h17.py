"""
LeetCode 17. Letter Combinations of a Phone Number

题目描述：
给定一个仅包含数字2-9的字符串，返回所有它能表示的字母组合。答案可以按任意顺序返回。
给出数字到字母的映射如下（与电话按键相同）。注意1不对应任何字母。

示例：
digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]

数据范围：
- 0 <= digits.length <= 4
- digits[i] 是范围 ['2', '9'] 的一个数字
"""

class Solution:
    def letterCombinations(self, digits: str) -> list[str]:
        if not digits:
            return []
        
        mapping = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }
        
        result = []
        
        def backtrack(index, current):
            if index == len(digits):
                result.append(current)
                return
            
            for letter in mapping[digits[index]]:
                backtrack(index + 1, current + letter)
        
        backtrack(0, "")
        return result
