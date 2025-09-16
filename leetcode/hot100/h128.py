"""
LeetCode 128. Longest Consecutive Sequence

题目描述：
给定一个未排序的整数数组nums，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
请你设计并实现时间复杂度为O(n)的算法解决此问题。

示例：
nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是[1, 2, 3, 4]。它的长度为4。

数据范围：
- 0 <= nums.length <= 10^5
- -10^9 <= nums[i] <= 10^9
"""

class Solution:
    def longestConsecutive(self, nums: list[int]) -> int:
        if not nums:
            return 0
        
        num_set = set(nums)
        max_length = 0
        
        for num in num_set:
            # 只从序列的起始点开始计算
            if num - 1 not in num_set:
                current_num = num
                current_length = 1
                
                # 向右扩展序列
                while current_num + 1 in num_set:
                    current_num += 1
                    current_length += 1
                
                max_length = max(max_length, current_length)
        
        return max_length
