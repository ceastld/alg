"""
LeetCode 33. Search in Rotated Sorted Array

题目描述：
整数数组nums按升序排列，数组中的值互不相同。在传递给函数之前，nums在预先未知的某个下标k上进行了旋转。
给你旋转后的数组nums和一个整数target，如果nums中存在这个目标值target，则返回它的下标，否则返回-1。

示例：
nums = [4,5,6,7,0,1,2], target = 0
输出：4

数据范围：
- 1 <= nums.length <= 5000
- -10^4 <= nums[i] <= 10^4
- nums中的每个值都独一无二
- nums原来是一个升序排序的数组，在预先未知的某个下标进行了旋转
- -10^4 <= target <= 10^4
"""

class Solution:
    def search(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if nums[mid] == target:
                return mid
            
            # 判断哪一半是有序的
            if nums[left] <= nums[mid]:
                # 左半部分有序
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # 右半部分有序
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
