"""
205. 同构字符串 - 标准答案
"""
from typing import List


class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        """
        标准解法：双哈希表
        
        解题思路：
        1. 使用两个哈希表分别记录s到t和t到s的映射关系
        2. 遍历字符串，检查映射关系是否一致
        3. 如果发现冲突（一个字符映射到多个字符，或多个字符映射到同一个字符），返回False
        
        时间复杂度：O(n)
        空间复杂度：O(1) - 最多26个字符
        """
        if len(s) != len(t):
            return False
        
        s_to_t = {}
        t_to_s = {}
        
        for i in range(len(s)):
            char_s = s[i]
            char_t = t[i]
            
            # 检查s到t的映射
            if char_s in s_to_t:
                if s_to_t[char_s] != char_t:
                    return False
            else:
                s_to_t[char_s] = char_t
            
            # 检查t到s的映射
            if char_t in t_to_s:
                if t_to_s[char_t] != char_s:
                    return False
            else:
                t_to_s[char_t] = char_s
        
        return True


def main():
    """测试标准答案"""
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
