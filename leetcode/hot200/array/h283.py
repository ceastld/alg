"""
283. 移动零
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，
同时保持非零元素的相对顺序。

请注意 ，必须在不复制数组的情况下原地对数组进行操作。

题目链接：https://leetcode.cn/problems/move-zeroes/

示例 1:
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]

示例 2:
输入: nums = [0]
输出: [0]

提示：
- 1 <= nums.length <= 104
- -231 <= nums[i] <= 231 - 1
"""

from typing import List


class Solution:
    """
    283. 移动零
    双指针经典题目
    """

    def moveZeroes(self, nums: List[int]) -> None:
        """
        请在这里实现你的解法
        注意：必须原地修改数组，不能返回任何值
        """
        # TODO: 在这里实现你的解法
        left = 0
        for right in range(len(nums)):
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    nums_1 = [0, 1, 0, 3, 12]
    solution.moveZeroes(nums_1)
    assert nums_1 == [1, 3, 12, 0, 0]

    # 测试用例2
    nums_2 = [0]
    solution.moveZeroes(nums_2)
    assert nums_2 == [0]

    # 测试用例3
    nums_3 = [1, 2, 3, 4, 5]
    solution.moveZeroes(nums_3)
    assert nums_3 == [1, 2, 3, 4, 5]

    # 测试用例4
    nums_4 = [0, 0, 0, 1, 2, 3]
    solution.moveZeroes(nums_4)
    assert nums_4 == [1, 2, 3, 0, 0, 0]

    # 测试用例5
    nums_5 = [1, 0, 1, 0, 1]
    solution.moveZeroes(nums_5)
    assert nums_5 == [1, 1, 1, 0, 0]

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
