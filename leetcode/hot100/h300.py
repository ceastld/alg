"""
LeetCode 300. Longest Increasing Subsequence

题目描述：
给你一个整数数组nums，找到其中最长严格递增子序列的长度。
子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7]是数组[0,3,1,6,2,2,7]的子序列。

示例：
nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是[2,3,7,18]，因此长度为4。

数据范围：
- 1 <= nums.length <= 2500
- -10^4 <= nums[i] <= 10^4
"""

class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        import bisect
        
        tails = []
        for num in nums:
            pos = bisect.bisect_left(tails, num)
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        
        return len(tails)
