"""
LeetCode 55. Jump Game

题目描述：
给定一个非负整数数组nums，你最初位于数组的第一个下标。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
判断你是否能够到达最后一个下标。

示例：
nums = [2,3,1,1,4]
输出：true
解释：可以先跳1步从下标0到达下标1，然后再从下标1跳3步到达最后一个下标。

数据范围：
- 1 <= nums.length <= 10^4
- 0 <= nums[i] <= 10^5
"""

class Solution:
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        max_reach = 0  # 当前能到达的最远位置
        
        for i in range(len(nums)):
            # 如果当前位置超出了能到达的最远位置，说明无法到达
            if i > max_reach:
                return False
            
            # 更新能到达的最远位置
            max_reach = max(max_reach, i + nums[i])
            
            # 如果能到达最后一个位置，直接返回True
            if max_reach >= len(nums) - 1:
                return True
        
        return True
