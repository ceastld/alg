"""
150. 逆波兰表达式求值
给你一个字符串数组 tokens ，表示一个根据 逆波兰表示法 表示的算术表达式。

请你计算该表达式。返回一个表示表达式值的整数。

注意：
- 有效的算符为 '+'、'-'、'*' 和 '/' 。
- 每个操作数（运算对象）都可以是一个整数或者另一个表达式。
- 两个整数之间的除法总是 向零截断 。
- 不会有任何除数为零的有效表达式。
- 输入表示一个有效的逆波兰表达式。
- 答案及所有中间计算结果可以用 32 位 整数表示。

题目链接：https://leetcode.cn/problems/evaluate-reverse-polish-notation/

示例 1:
输入: tokens = ["2","1","+","3","*"]
输出: 9
解释: 该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9

示例 2:
输入: tokens = ["4","13","5","/","+"]
输出: 6
解释: 该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6

示例 3:
输入: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出: 22

提示：
- 1 <= tokens.length <= 10^4
- tokens[i] 是一个算符（"+"、"-"、"*" 或 "/"），或是在范围 [-200, 200] 内的整数
"""
from typing import List


class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        """
        请在这里实现你的解法
        """
        func = {'+': lambda a,b: a+b, '-': lambda a,b: a-b, '*': lambda a,b: a*b, '/': lambda a,b: int(a/b)}
        stack = []
        for token in tokens:
            if token in func:
                b = stack.pop()
                a = stack.pop()
                stack.append(func[token](a,b))
            else:
                stack.append(int(token))
        return stack[0]


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    tokens = ["2", "1", "+", "3", "*"]
    assert solution.evalRPN(tokens) == 9
    
    # 测试用例2
    tokens = ["4", "13", "5", "/", "+"]
    assert solution.evalRPN(tokens) == 6
    
    # 测试用例3
    tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
    assert solution.evalRPN(tokens) == 22
    
    # 测试用例4
    tokens = ["4", "3", "-"]
    assert solution.evalRPN(tokens) == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
