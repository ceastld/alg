"""
150. 逆波兰表达式求值 - 标准答案
"""
from typing import List


class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        """
        标准解法：栈
        
        解题思路：
        1. 使用栈存储操作数
        2. 遍历tokens，如果是数字则入栈
        3. 如果是操作符，则弹出两个操作数进行计算
        4. 将计算结果重新入栈
        5. 最后栈中剩余的元素就是结果
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        stack = []
        
        for token in tokens:
            if token in ['+', '-', '*', '/']:
                # 弹出两个操作数
                b = stack.pop()
                a = stack.pop()
                
                # 根据操作符进行计算
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    # 向零截断
                    result = int(a / b)
                
                # 将结果入栈
                stack.append(result)
            else:
                # 数字，直接入栈
                stack.append(int(token))
        
        return stack[0]


def main():
    """测试标准答案"""
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
