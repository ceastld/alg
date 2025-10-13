"""
377. 组合总和IV
给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。

题目数据保证答案符合 32 位带符号整数。

题目链接：https://leetcode.cn/problems/combination-sum-iv/

示例 1:
输入：nums = [1,2,3], target = 4
输出：7
解释：
所有可能的组合为：
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
请注意，顺序不同的序列被视作不同的组合。

示例 2:
输入：nums = [9], target = 3
输出：0

提示：
- 1 <= nums.length <= 200
- 1 <= nums[i] <= 1000
- nums 中的所有值 互不相同
- 1 <= target <= 1000
"""

from typing import List


class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        nums.sort()
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(1, target + 1):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i - num]
        return dp[target]

def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,2,3]
    target = 4
    result = solution.combinationSum4(nums, target)
    expected = 7
    assert result == expected
    
    # 测试用例2
    nums = [9]
    target = 3
    result = solution.combinationSum4(nums, target)
    expected = 0
    assert result == expected
    
    # 测试用例3
    nums = [1,2,3]
    target = 1
    result = solution.combinationSum4(nums, target)
    expected = 1
    assert result == expected
    
    # 测试用例4
    nums = [1,2,3]
    target = 2
    result = solution.combinationSum4(nums, target)
    expected = 2
    assert result == expected
    
    # 测试用例5
    nums = [1,2,3]
    target = 3
    result = solution.combinationSum4(nums, target)
    expected = 4
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
