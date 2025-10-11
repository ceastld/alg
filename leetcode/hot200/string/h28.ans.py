"""
28. 实现 strStr() - 标准答案
"""
from typing import List


class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        """
        标准解法：KMP算法
        
        解题思路：
        1. 构建next数组（部分匹配表），记录模式串中每个位置的最长公共前后缀长度
        2. 使用双指针进行匹配，当字符不匹配时，根据next数组回退
        3. 避免重复匹配，提高效率
        
        时间复杂度：O(n+m)，其中n是haystack长度，m是needle长度
        空间复杂度：O(m)
        """
        if not needle:
            return 0
        
        # 构建next数组（部分匹配表）
        def build_next(pattern):
            next_arr = [0] * len(pattern)
            j = 0  # 前缀指针
            for i in range(1, len(pattern)):
                while j > 0 and pattern[i] != pattern[j]:
                    j = next_arr[j - 1]
                if pattern[i] == pattern[j]:
                    j += 1
                next_arr[i] = j
            return next_arr
        
        next_arr = build_next(needle)
        j = 0  # needle的指针
        
        for i in range(len(haystack)):
            # 当字符不匹配时，根据next数组回退
            while j > 0 and haystack[i] != needle[j]:
                j = next_arr[j - 1]
            
            # 字符匹配，j前进
            if haystack[i] == needle[j]:
                j += 1
            
            # 完全匹配，返回起始位置
            if j == len(needle):
                return i - j + 1
        
        return -1


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    haystack = "hello"
    needle = "ll"
    assert solution.strStr(haystack, needle) == 2
    
    # 测试用例2
    haystack = "aaaaa"
    needle = "bba"
    assert solution.strStr(haystack, needle) == -1
    
    # 测试用例3
    haystack = ""
    needle = ""
    assert solution.strStr(haystack, needle) == 0
    
    # 测试用例4
    haystack = "hello"
    needle = ""
    assert solution.strStr(haystack, needle) == 0
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
