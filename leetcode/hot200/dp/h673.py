"""
673. 最长递增子序列的个数
给定一个未排序的整数数组 nums ， 返回最长递增子序列的个数 。

注意 这个数列必须是 严格 递增的。

题目链接：https://leetcode.cn/problems/number-of-longest-increasing-subsequence/

示例 1:
输入: nums = [1,3,5,4,7]
输出: 2
解释: 有两个最长递增子序列，分别是 [1, 3, 4, 7] 和 [1, 3, 5, 7]。

示例 2:
输入: nums = [2,2,2,2,2]
输出: 5
解释: 最长递增子序列的长度是1，并且存在5个长度为1的子序列，每个子序列都是 [2]。

提示：
- 1 <= nums.length <= 2000
- -10^6 <= nums[i] <= 10^6
"""

from typing import List


class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1: return 1

        dp = [1] * n
        count = [1] * n
        max_length = 0
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    elif dp[j] + 1 == dp[i]:
                        count[i] += count[j]
            max_length = max(max_length, dp[i])

        res = 0
        for i in range(n):
            if dp[i] == max_length:
                res += count[i]
        return res


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,3,5,4,7]
    result = solution.findNumberOfLIS(nums)
    expected = 2
    assert result == expected
    
    # 测试用例2
    nums = [2,2,2,2,2]
    result = solution.findNumberOfLIS(nums)
    expected = 5
    assert result == expected
    
    # 测试用例3
    nums = [1]
    result = solution.findNumberOfLIS(nums)
    expected = 1
    assert result == expected
    
    # 测试用例4
    nums = [1,2,3,4,5]
    result = solution.findNumberOfLIS(nums)
    expected = 1
    assert result == expected
    
    # 测试用例5
    nums = [5,4,3,2,1]
    result = solution.findNumberOfLIS(nums)
    expected = 5
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
