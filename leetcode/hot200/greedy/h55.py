"""
55. 跳跃游戏
给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

题目链接：https://leetcode.cn/problems/jump-game/

示例 1:
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1，然后再从下标 1 跳 3 步到达最后一个下标。

示例 2:
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。

提示：
- 1 <= nums.length <= 3 * 10^4
- 0 <= nums[i] <= 10^5
"""

from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:

        max_reach = 0
        for i in range(len(nums)):
            if i > max_reach:
                return False
            max_reach = max(max_reach, i + nums[i])
        return True


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    assert solution.canJump([2, 3, 1, 1, 4]) == True

    # 测试用例2
    assert solution.canJump([3, 2, 1, 0, 4]) == False

    # 测试用例3
    assert solution.canJump([0]) == True

    # 测试用例4
    assert solution.canJump([1, 0, 1, 0]) == False

    # 测试用例5
    assert solution.canJump([2, 0, 0]) == True

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
