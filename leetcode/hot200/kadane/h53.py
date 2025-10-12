"""
53. 最大子数组和
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

子数组 是数组中的一个连续部分。

题目链接：https://leetcode.cn/problems/maximum-subarray/

示例 1:
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

示例 2:
输入：nums = [1]
输出：1

示例 3:
输入：nums = [5,4,-1,7,8]
输出：23

提示：
- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
"""

from typing import List


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    result = solution.maxSubArray(nums)
    expected = 6
    assert result == expected
    
    # 测试用例2
    nums = [1]
    result = solution.maxSubArray(nums)
    expected = 1
    assert result == expected
    
    # 测试用例3
    nums = [5,4,-1,7,8]
    result = solution.maxSubArray(nums)
    expected = 23
    assert result == expected
    
    # 测试用例4
    nums = [-1]
    result = solution.maxSubArray(nums)
    expected = -1
    assert result == expected
    
    # 测试用例5
    nums = [-2,-1]
    result = solution.maxSubArray(nums)
    expected = -1
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
