"""
1365. 有多少小于当前数字的数字
给你一个数组 nums，对于其中每个元素 nums[i]，请你统计数组中比它小的所有数字的数目。

换而言之，对于每个 nums[i] 你必须计算出有效的 j 的数量，其中 j 满足 j != i 且 nums[j] < nums[i]。

以数组形式返回答案。

题目链接：https://leetcode.cn/problems/how-many-numbers-are-smaller-than-the-current-number/

示例 1:
输入: nums = [8,1,2,2,3]
输出: [4,0,1,1,3]
解释:
对于 nums[0]=8 存在四个比它小的数字:(1,2,2,3)
对于 nums[1]=1 不存在比它小的数字
对于 nums[2]=2 存在一个比它小的数字:(1)
对于 nums[3]=2 存在一个比它小的数字:(1)
对于 nums[4]=3 存在三个比它小的数字:(1,2,2)

示例 2:
输入: nums = [6,5,4,8]
输出: [2,1,0,3]

提示：
- 2 <= nums.length <= 500
- 0 <= nums[i] <= 100
"""

from typing import List
from collections import Counter


class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        """
        请在这里实现你的解法
        """
        arr = sorted(nums)
        d = {}
        left = 0
        for i in range(len(arr)):
            if i > 0 and arr[i] != arr[i - 1]:
                left = i
            d[arr[i]] = left
        return [d[num] for num in nums]


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    nums = [8, 1, 2, 2, 3]
    expected = [4, 0, 1, 1, 3]
    assert solution.smallerNumbersThanCurrent(nums) == expected

    # 测试用例2
    nums = [6, 5, 4, 8]
    expected = [2, 1, 0, 3]
    assert solution.smallerNumbersThanCurrent(nums) == expected

    # 测试用例3
    nums = [7, 7, 7, 7]
    expected = [0, 0, 0, 0]
    assert solution.smallerNumbersThanCurrent(nums) == expected

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
