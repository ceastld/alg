"""
383. 赎金信
给你两个字符串：ransomNote 和 magazine，判断 ransomNote 能不能由 magazine 里面的字符构成。

如果可以，返回 true ；否则返回 false。

magazine 中的每个字符只能在 ransomNote 中使用一次。

题目链接：https://leetcode.cn/problems/ransom-note/

示例 1:
输入: ransomNote = "a", magazine = "b"
输出: false

示例 2:
输入: ransomNote = "aa", magazine = "ab"
输出: false

示例 3:
输入: ransomNote = "aa", magazine = "aab"
输出: true

提示：
- 1 <= ransomNote.length, magazine.length <= 10^5
- ransomNote 和 magazine 由小写英文字母组成
"""
from collections import Counter
from typing import List


class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        """
        请在这里实现你的解法
        """
        if len(ransomNote) > len(magazine):
            return False
        count = Counter(magazine)
        for char in ransomNote:
            if char not in count:
                return False
            count[char] -= 1
            if count[char] < 0:
                return False
        return True


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.canConstruct("a", "b") == False
    
    # 测试用例2
    assert solution.canConstruct("aa", "ab") == False
    
    # 测试用例3
    assert solution.canConstruct("aa", "aab") == True
    
    # 测试用例4
    assert solution.canConstruct("", "") == True
    
    # 测试用例5
    assert solution.canConstruct("a", "") == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
