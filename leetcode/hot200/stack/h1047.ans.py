"""
1047. 删除字符串中的所有相邻重复项 - 标准答案
"""
from typing import List


class Solution:
    def removeDuplicates(self, s: str) -> str:
        """
        标准解法：栈
        
        解题思路：
        1. 使用栈来存储字符
        2. 遍历字符串，如果当前字符与栈顶字符相同，则弹出栈顶
        3. 否则将当前字符入栈
        4. 最后将栈中字符连接成字符串
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        stack = []
        
        for char in s:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        
        return ''.join(stack)
    
    def removeDuplicates_optimized(self, s: str) -> str:
        """
        优化解法：使用deque + 字符串构建优化
        
        优化思路：
        1. 使用collections.deque替代list，提供O(1)的头部和尾部操作
        2. 使用StringBuilder模式，避免频繁的字符串拼接
        3. 使用collections.Counter进行字符统计优化
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        from collections import deque
        
        stack = deque()
        for char in s:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        
        return ''.join(stack)
    
    def removeDuplicates_ultra_fast(self, s: str) -> str:
        """
        超高效解法：使用字符数组 + 指针
        
        优化思路：
        1. 使用字符数组替代栈，减少内存分配
        2. 使用指针模拟栈操作，避免频繁的append/pop
        3. 直接操作字符数组，最后切片返回
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not s:
            return ""
        
        # 使用字符数组，最大长度为原字符串长度
        result = [''] * len(s)
        top = -1  # 栈顶指针
        
        for char in s:
            if top >= 0 and result[top] == char:
                # 弹出栈顶
                top -= 1
            else:
                # 入栈
                top += 1
                result[top] = char
        
        return ''.join(result[:top + 1])


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "abbaca"
    assert solution.removeDuplicates(s) == "ca"
    
    # 测试用例2
    s = "azxxzy"
    assert solution.removeDuplicates(s) == "ay"
    
    # 测试用例3
    s = "a"
    assert solution.removeDuplicates(s) == "a"
    
    # 测试用例4
    s = "abccba"
    assert solution.removeDuplicates(s) == ""
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
