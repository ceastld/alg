"""
724. 寻找数组的中心索引
给你一个整数数组 nums，请编写一个能够返回数组"中心索引"的方法。

数组中心索引是数组的一个索引，其左侧所有元素相加的和等于右侧所有元素相加的和。

如果数组不存在中心索引，返回 -1。如果数组有多个中心索引，应该返回最靠近左边的那一个。

注意：中心索引可能出现在数组的两端。

题目链接：https://leetcode.cn/problems/find-pivot-index/

示例 1:
输入: nums = [1, 7, 3, 6, 5, 6]
输出: 3
解释: 索引3 (nums[3] = 6) 的左侧数之和 (1 + 7 + 3 = 11)，与右侧数之和 (5 + 6 = 11) 相等。

示例 2:
输入: nums = [1, 2, 3]
输出: -1
解释: 数组中不存在满足此条件的中心索引。

示例 3:
输入: nums = [2, 1, -1]
输出: 0
解释: 索引0左侧不存在元素，视作和为0；右侧数之和为1 + (-1) = 0。

提示：
- 1 <= nums.length <= 10^4
- -1000 <= nums[i] <= 1000
"""

from typing import List


class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        """
        请在这里实现你的解法
        """
        s = sum(nums)
        left_sum = 0
        for i in range(len(nums)):
            if left_sum == s - left_sum - nums[i]:
                return i
            left_sum += nums[i]
        return -1

def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    nums = [1, 7, 3, 6, 5, 6]
    assert solution.pivotIndex(nums) == 3

    # 测试用例2
    nums = [1, 2, 3]
    assert solution.pivotIndex(nums) == -1

    # 测试用例3
    nums = [2, 1, -1]
    assert solution.pivotIndex(nums) == 0

    # 测试用例4
    nums = [1, 2, 3, 4, 5]
    assert solution.pivotIndex(nums) == -1

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
