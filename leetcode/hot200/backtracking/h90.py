"""
90. 子集II
给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。

题目链接：https://leetcode.cn/problems/subsets-ii/

示例 1:
输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]

示例 2:
输入：nums = [0]
输出：[[],[0]]

提示：
- 1 <= nums.length <= 10
- -10 <= nums[i] <= 10
"""

from typing import List


class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()

        def dfs(start, path):
            if start == len(nums):
                result.append(path)
                return
            j = start + 1
            while j < len(nums) and nums[j] == nums[start]:
                j += 1
            for c in range(j - start + 1):
                dfs(j, path + [nums[start]] * c)

        result = []
        dfs(0, [])
        return result


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    nums = [1, 2, 2]
    result = solution.subsetsWithDup(nums)
    expected = [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
    assert len(result) == len(expected)

    # 测试用例2
    nums = [0]
    result = solution.subsetsWithDup(nums)
    expected = [[], [0]]
    assert result == expected

    # 测试用例3
    nums = [1, 1, 2]
    result = solution.subsetsWithDup(nums)
    expected = [[], [1], [1, 1], [1, 1, 2], [1, 2], [2]]
    assert len(result) == len(expected)

    # 测试用例4
    nums = [1, 2, 3]
    result = solution.subsetsWithDup(nums)
    expected = [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
    assert len(result) == len(expected)

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
