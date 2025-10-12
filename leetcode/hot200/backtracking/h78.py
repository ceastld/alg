"""
78. 子集
给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

题目链接：https://leetcode.cn/problems/subsets/

示例 1:
输入: nums = [1,2,3]
输出: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

示例 2:
输入: nums = [0]
输出: [[],[0]]

提示：
- 1 <= nums.length <= 10
- -10 <= nums[i] <= 10
- nums 中的所有整数 互不相同
"""
from typing import List


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        请在这里实现你的解法
        """
        def dfs(i, path):
            if i == len(nums):
                result.append(path.copy())
                return
            dfs(i+1, path)
            dfs(i+1, path + [nums[i]])
        result = []
        dfs(0, [])
        return result


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 2, 3]
    result = solution.subsets(nums)
    expected = [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
    assert len(result) == len(expected)
    
    # 测试用例2
    nums = [0]
    result = solution.subsets(nums)
    expected = [[],[0]]
    assert len(result) == len(expected)
    
    # 测试用例3
    nums = [1, 2]
    result = solution.subsets(nums)
    expected = [[],[1],[2],[1,2]]
    assert len(result) == len(expected)
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
