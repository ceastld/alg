"""
459. 重复的子字符串 - 标准答案
"""
from typing import List


class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        """
        标准解法：字符串拼接
        
        解题思路：
        1. 将字符串s拼接两次：s + s
        2. 去掉拼接后字符串的首尾字符
        3. 如果原字符串s是重复子串构成，那么s一定在拼接后的字符串中
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        return s in (s + s)[1:-1]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "abab"
    assert solution.repeatedSubstringPattern(s) == True
    
    # 测试用例2
    s = "aba"
    assert solution.repeatedSubstringPattern(s) == False
    
    # 测试用例3
    s = "abcabcabcabc"
    assert solution.repeatedSubstringPattern(s) == True
    
    # 测试用例4
    s = "a"
    assert solution.repeatedSubstringPattern(s) == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
