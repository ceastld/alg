"""
494. 目标和
给你一个整数数组 nums 和一个整数 target 。

向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：

例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。

题目链接：https://leetcode.cn/problems/target-sum/

示例 1:
输入：nums = [1,1,1,1,1], target = 3
输出：5
解释：一共有 5 种方法让最终目标和为 3 。
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3

示例 2:
输入：nums = [1], target = 1
输出：1

提示：
- 1 <= nums.length <= 20
- 0 <= nums[i] <= 1000
- 0 <= sum(nums[i]) <= 1000
- -1000 <= target <= 1000
"""

from typing import List


class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,1,1,1,1]
    target = 3
    result = solution.findTargetSumWays(nums, target)
    expected = 5
    assert result == expected
    
    # 测试用例2
    nums = [1]
    target = 1
    result = solution.findTargetSumWays(nums, target)
    expected = 1
    assert result == expected
    
    # 测试用例3
    nums = [1,1,1,1,1]
    target = 5
    result = solution.findTargetSumWays(nums, target)
    expected = 1
    assert result == expected
    
    # 测试用例4
    nums = [1,1,1,1,1]
    target = 0
    result = solution.findTargetSumWays(nums, target)
    expected = 0
    assert result == expected
    
    # 测试用例5
    nums = [1,2,3,4,5]
    target = 3
    result = solution.findTargetSumWays(nums, target)
    expected = 3
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
