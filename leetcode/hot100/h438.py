"""
LeetCode 438. Find All Anagrams in a String

题目描述：
给定两个字符串s和p，找到s中所有p的异位词的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
异位词指由相同字母重排列形成的字符串（包括相同的字符串）。

示例：
s = "cbaebabacd", p = "abc"
输出：[0,6]
解释：起始索引等于0的子串是"cba"，它是"abc"的异位词。起始索引等于6的子串是"bac"，它是"abc"的异位词。

数据范围：
- 1 <= s.length, p.length <= 3 * 10^4
- s和p仅包含小写字母
"""

class Solution:
    def findAnagrams(self, s: str, p: str) -> list[int]:
        if len(s) < len(p):
            return []
        
        result = []
        p_count = [0] * 26
        s_count = [0] * 26
        
        # 统计p中每个字符的频次
        for char in p:
            p_count[ord(char) - ord('a')] += 1
        
        # 滑动窗口
        for i in range(len(s)):
            # 添加新字符
            s_count[ord(s[i]) - ord('a')] += 1
            
            # 移除窗口外的字符
            if i >= len(p):
                s_count[ord(s[i - len(p)]) - ord('a')] -= 1
            
            # 检查是否匹配
            if s_count == p_count:
                result.append(i - len(p) + 1)
        
        return result
