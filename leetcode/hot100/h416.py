"""
LeetCode 416. Partition Equal Subset Sum

题目描述：
给你一个只包含正整数的非空数组nums。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

示例：
nums = [1,5,11,5]
输出：true
解释：数组可以分割成[1, 5, 5]和[11]。

数据范围：
- 1 <= nums.length <= 200
- 1 <= nums[i] <= 100
"""

class Solution:
    def canPartition(self, nums: list[int]) -> bool:
        total = sum(nums)
        if total % 2 == 1:
            return False
        
        target = total // 2
        dp = [False] * (target + 1)
        dp[0] = True
        
        for num in nums:
            for i in range(target, num - 1, -1):
                dp[i] = dp[i] or dp[i - num]
        
        return dp[target]
