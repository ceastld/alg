"""
40. 组合总和II
给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用 一次 。

注意：解集不能包含重复的组合。

题目链接：https://leetcode.cn/problems/combination-sum-ii/

示例 1:
输入: candidates = [10,1,2,7,6,1,5], target = 8
输出:
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]

示例 2:
输入: candidates = [2,5,2,1,2], target = 5
输出:
[
[1,2,2],
[5]
]

提示:
- 1 <= candidates.length <= 100
- 1 <= candidates[i] <= 50
- 1 <= target <= 30
"""

from typing import List


class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        def dfs(start, path, remaining):
            # print(path,remaining) 
            if remaining == 0:
                result.append(path)
                return
            for i in range(start, len(candidates)):
                if candidates[i] > remaining:
                    break
                if i > start and candidates[i] == candidates[i-1]:
                    continue
                dfs(i+1, path+[candidates[i]], remaining - candidates[i])
        result = []
        dfs(0, [], target)
        return result


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    candidates = [10,1,2,7,6,1,5]
    target = 8
    result = solution.combinationSum2(candidates, target)
    expected = [[1,1,6],[1,2,5],[1,7],[2,6]]
    assert len(result) == len(expected)
    
    # 测试用例2
    candidates = [2,5,2,1,2]
    target = 5
    result = solution.combinationSum2(candidates, target)
    expected = [[1,2,2],[5]]
    assert len(result) == len(expected)
    
    # 测试用例3
    candidates = [1,1,1,1,1]
    target = 3
    result = solution.combinationSum2(candidates, target)
    expected = [[1,1,1]]
    assert len(result) == len(expected)
    
    # 测试用例4
    candidates = [1,2,3,4,5]
    target = 8
    result = solution.combinationSum2(candidates, target)
    expected = [[1,2,5],[1,3,4],[2,3,3]]
    assert len(result) == len(expected)
    
    assert solution.combinationSum2([10,1,2,7,6,1,5], 8) == [[1,1,6],[1,2,5],[1,7],[2,6]]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
