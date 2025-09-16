"""
LeetCode 238. Product of Array Except Self

题目描述：
给你一个整数数组nums，返回数组answer，其中answer[i]等于nums中除nums[i]之外其余各元素的乘积。
题目数据保证数组nums之中任意元素的全部前缀元素和后缀的乘积都在32位整数范围内。
请不要使用除法，且在O(n)时间复杂度内完成此题。

示例：
nums = [1,2,3,4]
输出：[24,12,8,6]

数据范围：
- 2 <= nums.length <= 10^5
- -30 <= nums[i] <= 30
- 保证数组nums之中任意元素的全部前缀元素和后缀的乘积都在32位整数范围内
"""

from typing import List

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # 非零元素乘积，
        # 零元素个数，如果 >=0
        prod = 1
        zero_count = 0
        for n in nums:
            if n != 0:
                prod *= n
            else:
                zero_count += 1
        if zero_count >= 2:
            return [0] * len(nums)
        elif zero_count == 1:
            return [prod if n == 0 else 0 for n in nums]
        else:
            return [prod if n == 0 else prod // n for n in nums]