"""
28. 实现 strStr()
给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。

说明：
当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与 C 语言的 strstr() 以及 Java 的 indexOf() 定义相符。

题目链接：https://leetcode.cn/problems/implement-strstr/

示例 1:
输入: haystack = "hello", needle = "ll"
输出: 2

示例 2:
输入: haystack = "aaaaa", needle = "bba"
输出: -1

示例 3:
输入: haystack = "", needle = ""
输出: 0

提示：
- 0 <= haystack.length, needle.length <= 5 * 10^4
- haystack 和 needle 仅由小写英文字符组成
"""
from typing import List


class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        """
        KMP算法实现
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
    """测试用例"""
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
