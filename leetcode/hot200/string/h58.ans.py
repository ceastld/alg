"""
剑指Offer58-II. 左旋转字符串 - 标准答案
"""
from typing import List


class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        """
        标准解法：字符串切片
        
        解题思路：
        1. 将字符串分为两部分：前n个字符和后面的字符
        2. 将两部分交换位置
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        return s[n:] + s[:n]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "abcdefg"
    n = 2
    assert solution.reverseLeftWords(s, n) == "cdefgab"
    
    # 测试用例2
    s = "lrloseumgh"
    n = 6
    assert solution.reverseLeftWords(s, n) == "umghlrlose"
    
    # 测试用例3
    s = "hello"
    n = 1
    assert solution.reverseLeftWords(s, n) == "elloh"
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
