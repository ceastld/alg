"""
LeetCode 448. Find All Numbers Disappeared in an Array

题目描述：
给你一个含n个整数的数组nums，其中nums[i]在区间[1, n]内。请你找出所有在[1, n]范围内但没有出现在nums中的数字，并以数组的形式返回结果。

示例：
nums = [4,3,2,7,8,2,3,1]
输出：[5,6]

数据范围：
- n == nums.length
- 1 <= n <= 10^5
- 1 <= nums[i] <= n
"""

class Solution:
    def findDisappearedNumbers(self, nums: list[int]) -> list[int]:
        # 使用数组本身作为哈希表
        for num in nums:
            index = abs(num) - 1
            if nums[index] > 0:
                nums[index] = -nums[index]
        
        # 收集所有正数位置的索引+1
        result = []
        for i in range(len(nums)):
            if nums[i] > 0:
                result.append(i + 1)
        
        return result
