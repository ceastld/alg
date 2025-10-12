"""
491. 递增子序列
给你一个整数数组 nums ，找出并返回所有该数组中不同的递增子序列，递增子序列中 至少有两个元素 。你可以按 任意顺序 返回答案。

数组中可能含有重复元素，如出现两个整数相等，也可以视作递增序列的一种特殊情况。

题目链接：https://leetcode.cn/problems/non-decreasing-subsequences/

示例 1:
输入：nums = [4,6,7,7]
输出：[[4,6],[4,6,7],[4,6,7,7],[4,7],[4,7,7],[6,7],[6,7,7],[7,7]]

示例 2:
输入：nums = [4,4,3,2,1]
输出：[[4,4]]

提示：
- 1 <= nums.length <= 15
- -100 <= nums[i] <= 100
- nums 中的所有整数 互不相同
"""

from typing import List


class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        def dfs(start, path):
            if len(path) > 1:
                result.append(path)
            if start == len(nums):
                return
            used = set()
            for i in range(start, len(nums)):
                if path and nums[i] < path[-1]:
                    continue
                if nums[i] not in used:
                    used.add(nums[i])
                    dfs(i + 1, path + [nums[i]])

        result = []
        dfs(0, [])
        return result


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    nums = [4, 6, 7, 7]
    result = solution.findSubsequences(nums)
    expected = [[4, 6], [4, 6, 7], [4, 6, 7, 7], [4, 7], [4, 7, 7], [6, 7], [6, 7, 7], [7, 7]]
    assert len(result) == len(expected)

    # 测试用例2
    nums = [4, 4, 3, 2, 1]
    result = solution.findSubsequences(nums)
    expected = [[4, 4]]
    assert result == expected

    # 测试用例3
    nums = [1, 2, 3, 4]
    result = solution.findSubsequences(nums)
    expected = [[1, 2], [1, 3], [1, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [1, 2, 3, 4], [2, 3], [2, 4], [2, 3, 4], [3, 4]]
    assert len(result) == len(expected)

    # 测试用例4
    nums = [1, 1, 1]
    result = solution.findSubsequences(nums)
    expected = [[1, 1], [1, 1, 1]]
    assert result == expected

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
