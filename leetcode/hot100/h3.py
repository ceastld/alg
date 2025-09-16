"""
LeetCode 3. Longest Substring Without Repeating Characters

题目描述：
给定一个字符串s，找出其中不含有重复字符的最长子串的长度。

示例：
s = "abcabcbb"
输出：3
解释：无重复字符的最长子串是 "abc"，长度为3

数据范围：
- 0 <= s.length <= 5 * 10^4
- s 由英文字母、数字、符号和空格组成
"""

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_set = set()
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            # 如果字符已存在，移动左指针
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            
            # 添加当前字符
            char_set.add(s[right])
            max_length = max(max_length, right - left + 1)
        
        return max_length
