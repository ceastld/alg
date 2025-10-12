"""
918. 环形子数组的最大和
给定一个长度为 n 的环形整数数组 nums ，返回 nums 的非空 子数组 的最大可能和 。

环形数组 意味着数组的末端将会与开头相连呈环状。形式上， nums[i] 的下一个元素是 nums[(i + 1) % n] ， nums[i] 的前一个元素是 nums[(i - 1 + n) % n] 。

子数组 最多只能包含固定缓冲区 nums 中的每个元素一次。形式上，对于子数组 nums[i], nums[i + 1], ..., nums[j] ，不存在 i <= k1, k2 <= j 其中 k1 % n == k2 % n 。

题目链接：https://leetcode.cn/problems/maximum-sum-circular-subarray/

示例 1:
输入：nums = [1,-2,3,-2]
输出：3
解释：从子数组 [3] 得到最大和 3

示例 2:
输入：nums = [5,-3,5]
输出：10
解释：从子数组 [5,5] 得到最大和 5 + 5 = 10

示例 3:
输入：nums = [-3,-2,-3]
输出：-2
解释：从子数组 [-2] 得到最大和 -2

提示：
- n == nums.length
- 1 <= n <= 3 * 10^4
- -3 * 10^4 <= nums[i] <= 3 * 10^4
"""

from typing import List


class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        sma = 0
        smi = 0
        total = 0
        max_sum = float("-inf")
        min_sum = float("inf")
        for num in nums:
            sma = min(sma, total)
            smi = max(smi, total)
            total += num
            max_sum = max(max_sum, total - sma)
            min_sum = min(min_sum, total - smi)
        if sum(nums) == min_sum:
            return max_sum
        return max(max_sum, sum(nums) - min_sum)


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    nums = [1, -2, 3, -2]
    result = solution.maxSubarraySumCircular(nums)
    expected = 3
    assert result == expected

    # 测试用例2
    nums = [5, -3, 5]
    result = solution.maxSubarraySumCircular(nums)
    expected = 10
    assert result == expected

    # 测试用例3
    nums = [-3, -2, -3]
    result = solution.maxSubarraySumCircular(nums)
    expected = -2
    assert result == expected

    # 测试用例4
    nums = [1]
    result = solution.maxSubarraySumCircular(nums)
    expected = 1
    assert result == expected

    # 测试用例5
    nums = [1, 2, 3, 4, 5]
    result = solution.maxSubarraySumCircular(nums)
    expected = 15
    assert result == expected

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
