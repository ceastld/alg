"""
20. 有效的括号 - 标准答案
"""
from typing import List


class Solution:
    def isValid(self, s: str) -> bool:
        """
        标准解法：栈
        
        解题思路：
        1. 使用栈来存储左括号
        2. 遇到左括号时入栈
        3. 遇到右括号时，检查栈顶是否匹配
        4. 最后检查栈是否为空
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if len(s) % 2 == 1:
            return False
        
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in mapping:
                # 遇到右括号
                if not stack or stack.pop() != mapping[char]:
                    return False
            else:
                # 遇到左括号
                stack.append(char)
        
        return not stack


def main():
    """测试标准答案"""
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
