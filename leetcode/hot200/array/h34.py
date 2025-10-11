"""
34. 在排序数组中查找元素的第一个和最后一个位置
给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。

题目链接：https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/

示例 1:
输入: nums = [5,7,7,8,8,10], target = 8
输出: [3,4]

示例 2:
输入: nums = [5,7,7,8,8,10], target = 6
输出: [-1,-1]

示例 3:
输入: nums = [], target = 0
输出: [-1,-1]

提示：
- 0 <= nums.length <= 10^5
- -10^9 <= nums[i] <= 10^9
- -10^9 <= target <= 10^9
"""
from typing import List


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        请在这里实现你的解法
        """
        if not nums:
            return [-1, -1]
        
        def lower_bound(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return left
        start = lower_bound(nums, target)
        if start == len(nums) or nums[start] != target:
            return [-1, -1]
        end = lower_bound(nums, target + 1) - 1
        return [start, end]

def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [5, 7, 7, 8, 8, 10]
    target = 8
    assert solution.searchRange(nums, target) == [3, 4]
    
    # 测试用例2
    nums = [5, 7, 7, 8, 8, 10]
    target = 6
    assert solution.searchRange(nums, target) == [-1, -1]
    
    # 测试用例3
    nums = []
    target = 0
    assert solution.searchRange(nums, target) == [-1, -1]
    
    # 测试用例4
    nums = [1]
    target = 1
    assert solution.searchRange(nums, target) == [0, 0]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
