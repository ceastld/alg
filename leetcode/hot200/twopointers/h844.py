"""
844. 比较含退格的字符串
给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 true 。# 代表退格字符。

注意：如果对空文本输入退格字符，文本继续为空。

题目链接：https://leetcode.cn/problems/backspace-string-compare/

示例 1:
输入: s = "ab#c", t = "ad#c"
输出: true
解释: s 和 t 都会变成 "ac"。

示例 2:
输入: s = "ab##", t = "c#d#"
输出: true
解释: s 和 t 都会变成 ""。

示例 3:
输入: s = "a#c", t = "b"
输出: false
解释: s 会变成 "c"，而 t 仍然是 "b"。

提示：
- 1 <= s.length, t.length <= 200
- s 和 t 只包含小写字母和字符 '#'
"""
from typing import List


class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        def convert(s: str) -> str:
            stack = []
            for char in s:
                if char != '#':
                    stack.append(char)
                elif stack:
                    stack.pop()
            return ''.join(stack)
        return convert(s) == convert(t)


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    s = "ab#c"
    t = "ad#c"
    assert solution.backspaceCompare(s, t) == True
    
    # 测试用例2
    s = "ab##"
    t = "c#d#"
    assert solution.backspaceCompare(s, t) == True
    
    # 测试用例3
    s = "a#c"
    t = "b"
    assert solution.backspaceCompare(s, t) == False
    
    # 测试用例4
    s = "a##c"
    t = "#a#c"
    assert solution.backspaceCompare(s, t) == True
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
