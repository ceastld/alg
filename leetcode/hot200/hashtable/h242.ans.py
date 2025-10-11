"""
242. 有效的字母异位词 - 标准答案
"""
from typing import Counter


class Solution:
    """
    242. 有效的字母异位词 - 标准解法
    """
    
    def isAnagram(self, s: str, t: str) -> bool:
        """
        标准解法：哈希表法
        
        解题思路：
        1. 统计两个字符串中每个字符的出现次数
        2. 比较两个字符串的字符频率是否相同
        3. 如果相同则互为字母异位词
        
        时间复杂度：O(n)
        空间复杂度：O(1) 因为只有26个小写字母
        """
        if len(s) != len(t):
            return False
        
        # 统计字符频率
        char_count = {}
        
        # 统计字符串s中每个字符的出现次数
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        # 统计字符串t中每个字符的出现次数
        for char in t:
            char_count[char] = char_count.get(char, 0) - 1
        
        # 检查所有字符的计数是否为0
        for count in char_count.values():
            if count != 0:
                return False
        
        return True
    
    def isAnagram_counter(self, s: str, t: str) -> bool:
        """
        使用Counter的解法
        
        解题思路：
        1. 使用Python的Counter类统计字符频率
        2. 直接比较两个Counter是否相等
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        return Counter(s) == Counter(t)
    
    def isAnagram_sort(self, s: str, t: str) -> bool:
        """
        排序法
        
        解题思路：
        1. 将两个字符串排序
        2. 比较排序后的字符串是否相等
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        return sorted(s) == sorted(t)
    
    def isAnagram_array(self, s: str, t: str) -> bool:
        """
        数组法（针对小写字母优化）
        
        解题思路：
        1. 使用长度为26的数组统计字符频率
        2. 遍历字符串s，增加对应字符的计数
        3. 遍历字符串t，减少对应字符的计数
        4. 检查数组是否全为0
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if len(s) != len(t):
            return False
        
        # 使用数组统计26个小写字母的频率
        count = [0] * 26
        
        for char in s:
            count[ord(char) - ord('a')] += 1
        
        for char in t:
            count[ord(char) - ord('a')] -= 1
        
        # 检查所有字符的计数是否为0
        for c in count:
            if c != 0:
                return False
        
        return True


def main():
    """测试标准答案"""
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
