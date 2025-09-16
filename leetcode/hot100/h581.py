"""
LeetCode 581. Shortest Unsorted Continuous Subarray

题目描述：
给你一个整数数组nums，你需要找出一个连续子数组，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。
请你找出符合题意的最短子数组，并输出它的长度。

示例：
nums = [2,6,4,8,10,9,15]
输出：5
解释：你只需要对[6, 4, 8, 10, 9]进行升序排序，那么整个表都会变为升序排序。

数据范围：
- 1 <= nums.length <= 10^4
- -10^5 <= nums[i] <= 10^5
"""

class Solution:
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        
        # 找到左边界：从左到右，找到第一个不满足递增的位置
        left = 0
        while left < n - 1 and nums[left] <= nums[left + 1]:
            left += 1
        
        if left == n - 1:
            return 0  # 数组已经有序
        
        # 找到右边界：从右到左，找到第一个不满足递增的位置
        right = n - 1
        while right > 0 and nums[right] >= nums[right - 1]:
            right -= 1
        
        # 找到中间部分的最小值和最大值
        min_val = min(nums[left:right + 1])
        max_val = max(nums[left:right + 1])
        
        # 扩展左边界
        while left > 0 and nums[left - 1] > min_val:
            left -= 1
        
        # 扩展右边界
        while right < n - 1 and nums[right + 1] < max_val:
            right += 1
        
        return right - left + 1
