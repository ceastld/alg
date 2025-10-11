"""
242. 有效的字母异位词
给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。

题目链接：https://leetcode.cn/problems/valid-anagram/

示例 1:
输入: s = "anagram", t = "nagaram"
输出: true

示例 2:
输入: s = "rat", t = "car"
输出: false

提示：
- 1 <= s.length, t.length <= 5 * 104
- s 和 t 仅包含小写字母
"""
from collections import Counter

class Solution:
    """
    242. 有效的字母异位词
    哈希表经典题目
    """
    
    def isAnagram(self, s: str, t: str) -> bool:
        """
        请在这里实现你的解法
        """
        if len(s) != len(t):
            return False
        count = [0] * 26
        for i in range(len(s)):
            count[ord(s[i]) - ord('a')] += 1
            count[ord(t[i]) - ord('a')] -= 1
        for i in range(26):
            if count[i] != 0:
                return False
        return True
    
    def isAnagram_counter(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        return Counter(s) == Counter(t)


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.isAnagram("anagram", "nagaram") == True
    
    # 测试用例2
    assert solution.isAnagram("rat", "car") == False
    
    # 测试用例3
    assert solution.isAnagram("a", "a") == True
    
    # 测试用例4
    assert solution.isAnagram("ab", "ba") == True
    
    # 测试用例5
    assert solution.isAnagram("abc", "def") == False
    
    # 测试用例6
    assert solution.isAnagram("listen", "silent") == True
    
    # 测试用例7
    assert solution.isAnagram("a", "ab") == False
    
    # 测试用例8
    assert solution.isAnagram("", "") == True
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
