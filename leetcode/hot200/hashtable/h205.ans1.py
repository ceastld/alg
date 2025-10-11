"""
205. 同构字符串 - 优化答案
"""
from typing import List


class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        """
        优化解法：使用zip和set
        
        解题思路：
        1. 使用zip将两个字符串配对
        2. 检查配对后的唯一性
        3. 如果s和t的字符都唯一，且配对也唯一，则同构
        
        时间复杂度：O(n)
        空间复杂度：O(1) - 最多26个字符
        """
        return len(set(zip(s, t))) == len(set(s)) == len(set(t))


def main():
    """测试优化答案"""
    solution = Solution()
    
    # 测试用例1
    s = "egg"
    t = "add"
    assert solution.isIsomorphic(s, t) == True
    
    # 测试用例2
    s = "foo"
    t = "bar"
    assert solution.isIsomorphic(s, t) == False
    
    # 测试用例3
    s = "paper"
    t = "title"
    assert solution.isIsomorphic(s, t) == True
    
    # 测试用例4
    s = "badc"
    t = "baba"
    assert solution.isIsomorphic(s, t) == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
