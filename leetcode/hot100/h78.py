"""
LeetCode 78. Subsets

题目描述：
给你一个整数数组nums，数组中的元素互不相同。返回该数组所有可能的子集（幂集）。
解集不能包含重复的子集。你可以按任意顺序返回解集。

示例：
nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

数据范围：
- 1 <= nums.length <= 10
- -10 <= nums[i] <= 10
- nums中的所有元素互不相同
"""

class Solution:
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        
        def backtrack(start, current):
            # 添加当前子集到结果中
            result.append(current[:])
            
            # 尝试添加剩余的元素
            for i in range(start, len(nums)):
                current.append(nums[i])
                backtrack(i + 1, current)
                current.pop()  # 回溯
        
        backtrack(0, [])
        return result
