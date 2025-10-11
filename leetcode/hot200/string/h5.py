"""
5. 最长回文子串
给你一个字符串 s，找到 s 中最长的回文子串。

题目链接：https://leetcode.cn/problems/longest-palindromic-substring/

示例 1:
输入: s = "babad"
输出: "bab"
解释: "aba" 同样是符合题意的答案。

示例 2:
输入: s = "cbbd"
输出: "bb"

示例 3:
输入: s = "a"
输出: "a"

提示：
- 1 <= s.length <= 1000
- s 仅由数字和英文字母组成
"""
from typing import List


class Solution:
    def longestPalindrome(self, s: str) -> str:
        """
        中心扩展法：统一处理奇数和偶数长度的回文
        
        算法思路：
        1. 遍历所有可能的回文中心（包括字符和字符间）
        2. 从中心向两边扩展，找到最长的回文
        3. 使用统一的循环处理奇数和偶数情况
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        n = len(s)
        if n == 0:
            return ""
        
        max_len = 0
        start = 0
        
        # 遍历所有可能的回文中心
        for i in range(2 * n):
            left = i // 2
            right = (i + 1) // 2
            
            # 从中心向两边扩展
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
            
            # 调整边界（因为while循环多扩展了一次）
            left += 1
            right -= 1
            
            # 计算当前回文长度
            current_len = right - left + 1
            
            # 更新最长回文
            if current_len > max_len:
                max_len = current_len
                start = left
        
        return s[start:start + max_len]


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    s = "babad"
    result = solution.longestPalindrome(s)
    assert result == "bab" or result == "aba"
    
    # 测试用例2
    s = "cbbd"
    assert solution.longestPalindrome(s) == "bb"
    
    # 测试用例3
    s = "a"
    assert solution.longestPalindrome(s) == "a"
    
    # 测试用例4
    s = "ac"
    result = solution.longestPalindrome(s)
    assert result == "a" or result == "c"
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
