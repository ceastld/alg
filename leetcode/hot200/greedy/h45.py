"""
45. 跳跃游戏II
给你一个非负整数数组 nums ，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

假设你总是可以到达数组的最后一个位置。

题目链接：https://leetcode.cn/problems/jump-game-ii/

示例 1:
输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1，跳 1 步，然后跳 3 步到达数组的最后一个位置。

示例 2:
输入: nums = [2,3,0,1,4]
输出: 2

提示：
- 1 <= nums.length <= 10^4
- 0 <= nums[i] <= 1000
- 题目保证可以到达 nums[n-1]
"""

from typing import List


class Solution:
    def jump(self, nums: List[int]) -> int:
        dp = [float('inf')] * len(nums)
        dp[0] = 0
        for i,jump in enumerate(nums):
            for j in range(i+1, min(i+jump+1, len(nums))):
                dp[j] = min(dp[j], dp[i] + 1)
            
        return dp[-1]
        


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.jump([2,3,1,1,4]) == 2
    
    # 测试用例2
    assert solution.jump([2,3,0,1,4]) == 2
    
    # 测试用例3
    assert solution.jump([1,2,3]) == 2
    
    # 测试用例4
    assert solution.jump([1]) == 0
    
    # 测试用例5
    assert solution.jump([2,1]) == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
