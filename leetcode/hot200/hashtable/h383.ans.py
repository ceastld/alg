"""
383. 赎金信 - 标准答案
"""
from typing import List
from collections import Counter


class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        """
        标准解法：哈希表计数
        
        解题思路：
        1. 统计magazine中每个字符的出现次数
        2. 遍历ransomNote中的每个字符，检查是否在magazine中有足够的字符
        3. 如果某个字符在ransomNote中需要的数量超过magazine中可用的数量，返回False
        4. 如果所有字符都满足条件，返回True
        
        时间复杂度：O(m + n)，其中m是magazine长度，n是ransomNote长度
        空间复杂度：O(1)，最多26个字符
        """
        # 统计magazine中每个字符的出现次数
        magazine_count = Counter(magazine)
        
        # 检查ransomNote中的每个字符
        for char in ransomNote:
            if magazine_count[char] <= 0:
                return False
            magazine_count[char] -= 1
        
        return True


def main():
    """测试标准答案"""
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
