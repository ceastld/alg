"""
LeetCode 394. Decode String

题目描述：
给定一个经过编码的字符串，返回它解码后的字符串。
编码规则为：k[encoded_string]，表示其中方括号内部的encoded_string正好重复k次。注意k保证为正整数。
你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数k。例如不会出现像3a或2[4]的输入。

示例：
s = "3[a]2[bc]"
输出："aaabcbc"

数据范围：
- 1 <= s.length <= 30
- s由小写英文字母、数字和方括号'[]'组成
- s是一个有效的编码字符串
- s中所有整数的取值范围为[1, 300]
"""

class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        current_string = ""
        current_num = 0
        
        for char in s:
            if char.isdigit():
                current_num = current_num * 10 + int(char)
            elif char == '[':
                stack.append((current_string, current_num))
                current_string = ""
                current_num = 0
            elif char == ']':
                prev_string, num = stack.pop()
                current_string = prev_string + current_string * num
            else:
                current_string += char
        
        return current_string
