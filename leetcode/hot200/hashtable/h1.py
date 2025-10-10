"""
1. 两数之和
给定一个整数数组 nums 和一个整数目标值 target，
请你在该数组中找出 和为目标值 target 的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

题目链接：https://leetcode.cn/problems/two-sum/

示例 1:
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

示例 2:
输入：nums = [3,2,4], target = 6
输出：[1,2]

示例 3:
输入：nums = [3,3], target = 6
输出：[0,1]

提示：
- 2 <= nums.length <= 104
- -109 <= nums[i] <= 109
- -109 <= target <= 109
- 只会存在一个有效答案
"""

from typing import List


class Solution:
    """
    1. 两数之和
    哈希表经典题目
    """

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        index_map = {}
        for i, num in enumerate(nums):
            if target - num in index_map:
                return [index_map[target - num], i]
            index_map[num] = i
        for k, v in index_map.items():
            if k < target / 2 and target - k in index_map:
                return [v, index_map[target - k]]
        return []


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    assert solution.twoSum([2, 7, 11, 15], 9) == [0, 1]

    # 测试用例2
    assert solution.twoSum([3, 2, 4], 6) == [1, 2]

    # 测试用例3
    assert solution.twoSum([3, 3], 6) == [0, 1]

    # 测试用例4
    assert solution.twoSum([-1, -2, -3, -4, -5], -8) == [2, 4]

    # 测试用例5
    assert solution.twoSum([0, 4, 3, 0], 0) == [0, 3]

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
