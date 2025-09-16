"""
LeetCode 39. Combination Sum

题目描述：
给你一个无重复元素的整数数组candidates和一个目标整数target，找出candidates中可以使数字和为target的所有不同组合。
candidates中的同一个数字可以无限制重复被选取。如果至少一个数字的被选数量不同，则两种组合是不同的。

示例：
candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]

数据范围：
- 1 <= candidates.length <= 30
- 1 <= candidates[i] <= 200
- candidate中的每个元素都互不相同
- 1 <= target <= 500
"""

class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        result = []
        
        def backtrack(start, current, remaining):
            if remaining == 0:
                result.append(current[:])
                return
            
            for i in range(start, len(candidates)):
                if candidates[i] <= remaining:
                    current.append(candidates[i])
                    backtrack(i, current, remaining - candidates[i])
                    current.pop()
        
        backtrack(0, [], target)
        return result
