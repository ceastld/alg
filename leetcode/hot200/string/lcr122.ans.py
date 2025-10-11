"""
LCR122. 路径加密 - 标准答案
"""
from typing import List


class Solution:
    def pathEncryption(self, path: str) -> str:
        """
        标准解法：字符串替换
        
        解题思路：
        1. 将路径中的 "." 替换为空格 " "
        2. 使用字符串的replace方法直接替换
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        return path.replace('.', ' ')


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    path = "a.aef.qerf.bb"
    assert solution.pathEncryption(path) == "a aef qerf bb"
    
    # 测试用例2
    path = "hello.world"
    assert solution.pathEncryption(path) == "hello world"
    
    # 测试用例3
    path = "a"
    assert solution.pathEncryption(path) == "a"
    
    # 测试用例4
    path = "a.b.c.d"
    assert solution.pathEncryption(path) == "a b c d"
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()