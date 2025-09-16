"""
LeetCode 34. Find First and Last Position of Element in Sorted Array

题目描述：
给定一个按照升序排列的整数数组nums，和一个目标值target。找出给定目标值在数组中的开始位置和结束位置。
如果数组中不存在目标值target，返回[-1, -1]。

示例：
nums = [5,7,7,8,8,10], target = 8
输出：[3,4]

数据范围：
- 0 <= nums.length <= 10^5
- -10^9 <= nums[i] <= 10^9
- nums是一个非递减数组
- -10^9 <= target <= 10^9
"""

class Solution:
    def searchRange(self, nums: list[int], target: int) -> list[int]:
        def find_first():
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return left if left < len(nums) and nums[left] == target else -1
        
        def find_last():
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] <= target:
                    left = mid + 1
                else:
                    right = mid - 1
            return right if right >= 0 and nums[right] == target else -1
        
        if not nums:
            return [-1, -1]
        
        first = find_first()
        if first == -1:
            return [-1, -1]
        
        return [first, find_last()]
