"""
39. 组合总和
给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为 target 的所有不同组合，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。

对于给定的输入，保证和为 target 的不同组合数少于 150 个。

题目链接：https://leetcode.cn/problems/combination-sum/

示例 1:
输入: candidates = [2,3,6,7], target = 7
输出: [[2,2,3],[7]]
解释: 2 和 3 可以无限制重复被选取，7 也是一个候选数。

示例 2:
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]

示例 3:
输入: candidates = [2], target = 1
输出: []

提示：
- 1 <= candidates.length <= 30
- 2 <= candidates[i] <= 40
- candidates 中的所有元素 互不相同
- 1 <= target <= 40
"""
from typing import List


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        请在这里实现你的解法
        """
        candidates.sort()
        def dfs(start, path, remaining):
            if remaining == 0:
                result.append(path.copy())
                return
            for i in range(start, len(candidates)):
                if candidates[i] <= remaining:
                    dfs(i, path+[candidates[i]], remaining - candidates[i])
                else:
                    break
        result = []
        dfs(0, [], target)
        return result
        

def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    candidates = [2, 3, 6, 7]
    target = 7
    result = solution.combinationSum(candidates, target)
    expected = [[2,2,3],[7]]
    assert len(result) == len(expected)
    
    # 测试用例2
    candidates = [2, 3, 5]
    target = 8
    result = solution.combinationSum(candidates, target)
    expected = [[2,2,2,2],[2,3,3],[3,5]]
    assert len(result) == len(expected)
    
    # 测试用例3
    candidates = [2]
    target = 1
    result = solution.combinationSum(candidates, target)
    expected = []
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
