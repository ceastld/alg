"""
LeetCode 76. Minimum Window Substring

题目描述：
给你一个字符串s、一个字符串t。返回s中涵盖t所有字符的最小子串。如果s中不存在涵盖t所有字符的子串，则返回空字符串""。
注意：
- 对于t中重复字符，我们寻找的子字符串中该字符数量必须不少于t中该字符数量
- 如果s中存在这样的子串，我们保证它是唯一的答案

示例：
s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串"BANC"包含来自字符串t的'A'、'B'和'C'。

数据范围：
- m == s.length
- n == t.length
- 1 <= m, n <= 10^5
- s和t由英文字母组成
"""

class Solution:
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if not s or not t:
            return ""
        
        # 统计t中每个字符的出现次数
        dict_t = {}
        for char in t:
            dict_t[char] = dict_t.get(char, 0) + 1
        
        required = len(dict_t)  # 需要匹配的字符种类数
        left = right = 0
        formed = 0  # 当前窗口中匹配的字符种类数
        
        window_counts = {}
        ans = float('inf'), None, None  # (窗口长度, 左边界, 右边界)
        
        while right < len(s):
            # 扩展右边界
            char = s[right]
            window_counts[char] = window_counts.get(char, 0) + 1
            
            if char in dict_t and window_counts[char] == dict_t[char]:
                formed += 1
            
            # 收缩左边界
            while left <= right and formed == required:
                char = s[left]
                
                # 更新答案
                if right - left + 1 < ans[0]:
                    ans = (right - left + 1, left, right)
                
                window_counts[char] -= 1
                if char in dict_t and window_counts[char] < dict_t[char]:
                    formed -= 1
                
                left += 1
            
            right += 1
        
        return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]
