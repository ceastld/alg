"""
LeetCode 283. Move Zeroes

题目描述：
给定一个数组nums，编写一个函数将所有0移动到数组的末尾，同时保持非零元素的相对顺序。
请注意，必须在不复制数组的情况下原地对数组进行操作。

示例：
nums = [0,1,0,3,12]
输出：[1,3,12,0,0]

数据范围：
- 1 <= nums.length <= 10^4
- -2^31 <= nums[i] <= 2^31 - 1
"""

class Solution:
    def moveZeroes(self, nums: list[int]) -> None:
        left = 0
        for right in range(len(nums)):
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
