"""
LeetCode 46. Permutations

题目描述：
给定一个不含重复数字的数组nums，返回其所有可能的全排列。你可以按任意顺序返回答案。

示例：
nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

数据范围：
- 1 <= nums.length <= 6
- -10 <= nums[i] <= 10
- nums中的所有整数互不相同
"""

class Solution:
    def permute(self, nums: list[int]) -> list[list[int]]:
        result = []
        
        def backtrack(current):
            if len(current) == len(nums):
                result.append(current[:])
                return
            
            for num in nums:
                if num not in current:
                    current.append(num)
                    backtrack(current)
                    current.pop()
        
        backtrack([])
        return result
