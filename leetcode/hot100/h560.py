"""
LeetCode 560. Subarray Sum Equals K

题目描述：
给你一个整数数组nums和一个整数k，请你统计并返回该数组中和为k的连续子数组的个数。

示例：
nums = [1,1,1], k = 2
输出：2

数据范围：
- 1 <= nums.length <= 2 * 10^4
- -1000 <= nums[i] <= 1000
- -10^7 <= k <= 10^7
"""

class Solution:
    def subarraySum(self, nums: list[int], k: int) -> int:
        from collections import defaultdict
        
        count = 0
        prefix_sum = 0
        sum_count = defaultdict(int)
        sum_count[0] = 1  # 前缀和为0出现1次
        
        for num in nums:
            prefix_sum += num
            # 如果存在前缀和为 prefix_sum - k，则存在子数组和为k
            if prefix_sum - k in sum_count:
                count += sum_count[prefix_sum - k]
            
            # 记录当前前缀和的出现次数
            sum_count[prefix_sum] += 1
        
        return count
