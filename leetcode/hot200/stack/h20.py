"""
20. 有效的括号
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：
1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

题目链接：https://leetcode.cn/problems/valid-parentheses/

示例 1:
输入: s = "()"
输出: true

示例 2:
输入: s = "()[]{}"
输出: true

示例 3:
输入: s = "(]"
输出: false

示例 4:
输入: s = "([)]"
输出: false

提示：
- 1 <= s.length <= 10^4
- s 仅由括号 '()[]{}' 组成
"""
from typing import List


class Solution:
    def isValid(self, s: str) -> bool:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    s = "()"
    assert solution.isValid(s) == True
    
    # 测试用例2
    s = "()[]{}"
    assert solution.isValid(s) == True
    
    # 测试用例3
    s = "(]"
    assert solution.isValid(s) == False
    
    # 测试用例4
    s = "([)]"
    assert solution.isValid(s) == False
    
    # 测试用例5
    s = "{[]}"
    assert solution.isValid(s) == True
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
