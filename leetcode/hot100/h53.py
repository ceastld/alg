"""
LeetCode 53. Maximum Subarray

题目描述：
给你一个整数数组nums，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
子数组是数组中的一个连续部分。

示例：
nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组[4,-1,2,1]的和最大，为6。

数据范围：
- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
"""

class Solution:
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # Kadane算法
        max_sum = current_sum = nums[0]
        
        for i in range(1, len(nums)):
            # 如果当前和小于0，重新开始计算
            current_sum = max(nums[i], current_sum + nums[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum
